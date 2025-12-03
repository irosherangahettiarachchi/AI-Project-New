import os
import argparse
import pandas as pd
import json
import math
from typing import List, Dict, TypedDict, Annotated
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(override=True)
# --- CONFIGURATION & LLM SETUP ---

# We implement the Multi-LLM Strategy here
def get_llm(role: str):
    """
    Factory to switch models based on agent role.
    Listing/Creative -> Llama3 (Better creativity)
    QA/Logic -> Mistral (Good at following strict instructions)
    """
    # if role == "listing":
    #     return ChatOllama(model="llama3", temperature=0.7)
    # elif role == "qa":
    #     return ChatOllama(model="mistral", temperature=0.1)
    # else:
    #     return ChatOllama(model="llama3", temperature=0)
    if role == "listing":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
)
    elif role == "qa":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            google_api_key=os.getenv("GOOGLE_API_KEY")
)
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=1.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
)
    

# --- STATE DEFINITION ---

class AgentState(TypedDict):
    """Global state passed between agents"""
    catalog_path: str
    orders_path: str
    output_dir: str
    
    # Data Flow
    raw_catalog: List[Dict]
    selected_skus: List[Dict]    # Output of Sourcing Agent
    listings: List[Dict]         # Output of Listing Agent
    listing_redlines: List[Dict] # Output of QA Agent
    price_updates: List[Dict]    # Output of Pricing Agent
    stock_updates: List[Dict]    # Output of Pricing Agent
    order_actions: List[Dict]    # Output of Routing Agent
    daily_report: str            # Output of Reporter Agent

# --- AGENT NODES ---

def sourcing_agent(state: AgentState):
    print("--- [1/6] Product Sourcing Agent ---")
    df = pd.read_csv(state['catalog_path'])
    
    # Criteria: Stock >= 10. 
    # Note: We calculate potential margin later, but here we filter basic viability.
    # Let's assume a base viability check first.
    filtered = df[df['stock'] >= 10].copy()
    
    # Pick top 10 based on stock level (simple heuristic for 'best' supply)
    top_10 = filtered.sort_values(by='stock', ascending=False).head(10)
    
    selected = top_10.to_dict(orient='records')
    
    # Save artifact
    with open(os.path.join(state['output_dir'], "selection.json"), "w") as f:
        json.dump(selected, f, indent=2)
        
    return {"selected_skus": selected, "raw_catalog": df.to_dict(orient='records')}

def pricing_agent(state: AgentState):
    print("--- [2/6] Pricing & Stock Agent ---")
    selected = state['selected_skus']
    price_updates = []
    stock_updates = []
    
    # Pricing Formula
    # P = (Cost + Shipping + Fee + GST) / (1 - Margin)
    # Fee = 2.9% * P + 0.30
    # GST = 10% * P (Assuming AU for safety, or 0 if US. Let's implement AU flag logic or worst case)
    # Let's solve for P algebraically:
    # P - (0.029P) - (0.10P) - 0.25P = Cost + Shipping + 0.30
    # P (1 - 0.029 - 0.10 - 0.25) = Cost + Shipping + 0.30
    # P (0.621) = Cost + Shipping + 0.30
    # P = (Cost + Shipping + 0.30) / 0.621
    
    divisor = 0.621 # 1 - 0.029 (fee) - 0.10 (GST) - 0.25 (Margin)
    
    for item in selected:
        cost = float(item['cost_price'])
        shipping = float(item['shipping_cost'])
        
        # Calculate Minimum Price
        min_price = (cost + shipping + 0.30) / divisor
        
        # Round up to nearest 0.50
        final_price = math.ceil(min_price * 2) / 2
        
        price_updates.append({
            "sku": item['supplier_sku'],
            "new_price": final_price,
            "cost_basis": cost + shipping
        })
        
        stock_updates.append({
            "sku": item['supplier_sku'],
            "stock_level": item['stock']
        })

    # Save artifacts
    pd.DataFrame(price_updates).to_csv(os.path.join(state['output_dir'], "price_update.csv"), index=False)
    pd.DataFrame(stock_updates).to_csv(os.path.join(state['output_dir'], "stock_update.csv"), index=False)
    
    return {"price_updates": price_updates, "stock_updates": stock_updates}

def listing_agent(state: AgentState):
    print("--- [3/6] Listing Agent (LLM) ---")
    llm = get_llm("listing")
    generated_listings = []
    
    # We process in batch or loop. Loop is safer for local LLM context limits.
    prompt_template = ChatPromptTemplate.from_template(
        """You are a professional Shopify Copywriter.
        Create a listing for the following product. 
        Output strictly JSON with keys: title, description_html, bullets (list), tags (list), seo_title, seo_description.
        
        Product Data:
        Name: {name}
        Category: {category}
        Description: {description}
        Features: Weight {weight_kg}kg
        """
    )
    
    chain = prompt_template | llm | JsonOutputParser()
    
    for item in state['selected_skus']:
        try:
            res = chain.invoke({
                "name": item['name'],
                "category": item['category'],
                "description": item['description'],
                "weight_kg": item['weight_kg']
            })
            res['sku'] = item['supplier_sku']
            generated_listings.append(res)
            print(f"Generated listing for {item['supplier_sku']}")
        except Exception as e:
            print(f"Failed to generate listing for {item['supplier_sku']}: {e}")

    with open(os.path.join(state['output_dir'], "listings.json"), "w") as f:
        json.dump(generated_listings, f, indent=2)
        
    return {"listings": generated_listings}

def qa_agent(state: AgentState):
    print("--- [4/6] QA Agent (LLM) ---")
    llm = get_llm("qa")
    listings = state['listings']
    redlines = []
    
    prompt = ChatPromptTemplate.from_template(
        """Review this Shopify listing for compliance. 
        Check for: Grammar errors, Over-promising (claims not in data), and SEO length.
        Output JSON: {{ "status": "PASS" or "FAIL", "issues": ["issue1", "issue2"] }}
        
        Listing: {listing}
        """
    )
    
    chain = prompt | llm | JsonOutputParser()
    
    for listing in listings:
        try:
            res = chain.invoke({"listing": json.dumps(listing)})
            res['sku'] = listing['sku']
            if res['status'] == "FAIL":
                redlines.append(res)
        except:
            pass # Skip malformed LLM responses
            
    with open(os.path.join(state['output_dir'], "listing_redlines.json"), "w") as f:
        json.dump(redlines, f, indent=2)

    return {"listing_redlines": redlines}

def order_routing_agent(state: AgentState):
    print("--- [5/6] Order Routing Agent ---")
    orders_df = pd.read_csv(state['orders_path'])
    raw_catalog = pd.DataFrame(state['raw_catalog'])
    
    actions = []
    llm = get_llm("qa") # Using the logic/mistral model for emails
    
    email_prompt = ChatPromptTemplate.from_template(
        "Write a short customer service email regarding order {order_id}. Context: {context}. Keep it professional."
    )
    email_chain = email_prompt | llm | StrOutputParser()
    
    for _, order in orders_df.iterrows():
        sku_data = raw_catalog[raw_catalog['supplier_sku'] == order['sku']]
        
        action = {
            "order_id": order['order_id'],
            "sku": order['sku'],
            "action": "UNKNOWN",
            "email_draft": ""
        }
        
        if sku_data.empty:
            action['action'] = "CANCEL_REFUND"
            context = "Item discontinued/not found."
        else:
            stock = sku_data.iloc[0]['stock']
            if stock >= order['quantity']:
                action['action'] = "FULFILL_DROPSHIP"
                context = "Order confirmed and shipping soon."
            else:
                action['action'] = "BACKORDER"
                context = f"Item temporarily out of stock. Expected delay: {sku_data.iloc[0]['supplier_lead_days']} days."
        
        # Generate Email
        action['email_draft'] = email_chain.invoke({"order_id": order['order_id'], "context": context})
        actions.append(action)
        
    with open(os.path.join(state['output_dir'], "order_actions.json"), "w") as f:
        json.dump(actions, f, indent=2)
        
    return {"order_actions": actions}

def reporter_agent(state: AgentState):
    print("--- [6/6] Reporter Agent ---")
    llm = get_llm("listing") # Use the creative model for the summary
    
    # Gather stats
    total_selected = len(state['selected_skus'])
    total_listings = len(state['listings'])
    qa_fails = len(state['listing_redlines'])
    orders_processed = len(state['order_actions'])
    
    summary_prompt = f"""
    Generate a Daily Operations Report in Markdown format.
    
    Stats:
    - SKUs Sourced: {total_selected}
    - Listings Generated: {total_listings}
    - QA Rejections: {qa_fails}
    - Orders Processed: {orders_processed}
    
    Listing Issues Found: {json.dumps(state['listing_redlines'])}
    
    Include a section "Executive Summary" and "Action Items".
    """
    
    report = llm.invoke(summary_prompt).content
    
    with open(os.path.join(state['output_dir'], "daily_report.md"), "w") as f:
        f.write(report)
        
    return {"daily_report": report}

# --- CLI & GRAPH CONSTRUCTION ---

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("sourcing", sourcing_agent)
    workflow.add_node("pricing", pricing_agent)
    workflow.add_node("listing", listing_agent)
    workflow.add_node("qa", qa_agent)
    workflow.add_node("routing", order_routing_agent)
    workflow.add_node("reporting", reporter_agent)
    
    # Define Edges (Sequential Workflow)
    workflow.set_entry_point("sourcing")
    workflow.add_edge("sourcing", "pricing")
    workflow.add_edge("pricing", "listing")
    workflow.add_edge("listing", "qa")
    workflow.add_edge("qa", "routing")
    workflow.add_edge("routing", "reporting")
    workflow.add_edge("reporting", END)
    
    return workflow.compile()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shopify Dropshipping Ops Agent")
    parser.add_argument("--catalog", required=True, help="Path to supplier CSV")
    parser.add_argument("--orders", required=True, help="Path to orders CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(args.out, exist_ok=True)
    
    # Initialize App
    app = build_graph()
    
    # Run
    print(f"Starting Ops Agent...\nInputs: {args.catalog}, {args.orders}\nOutput: {args.out}")
    initial_state = {
        "catalog_path": args.catalog,
        "orders_path": args.orders,
        "output_dir": args.out,
        "raw_catalog": [],
        "selected_skus": [],
        "listings": [],
        "listing_redlines": [],
        "price_updates": [],
        "stock_updates": [],
        "order_actions": [],
        "daily_report": ""
    }
    
    app.invoke(initial_state)
    print("\nWorklfow Complete. Check 'out/' directory.")