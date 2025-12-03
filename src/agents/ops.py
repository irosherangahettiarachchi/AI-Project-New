import os
import json
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import get_llm
from src.state import AgentState

def order_routing_agent(state: AgentState):
    print("--- [5/7] Order Routing Agent ---")
    orders_df = pd.read_csv(state['orders_path'])
    raw_catalog = pd.DataFrame(state['raw_catalog'])
    
    actions = []
    llm = get_llm("qa") 
    
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
        
        action['email_draft'] = email_chain.invoke({"order_id": order['order_id'], "context": context})
        actions.append(action)
        
    with open(os.path.join(state['output_dir'], "order_actions.json"), "w") as f:
        json.dump(actions, f, indent=2)
        
    return {"order_actions": actions}

def reporter_agent(state: AgentState):
    print("--- [6/7] Reporter Agent ---")
    llm = get_llm("listing")
    
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

def manager_agent(state: AgentState):
    print("--- [7/7] Manager Agent ---")
    llm = get_llm("manager")
    
    summary_prompt = f"""
    Review all outputs and provide high-level recommendations.
    Current State Summary:
    - SKUs Selected: {len(state['selected_skus'])}
    - Listings: {len(state['listings'])}
    - QA Failures: {len(state['listing_redlines'])}
    - Orders Processed: {len(state['order_actions'])}
    """
    
    report = llm.invoke(summary_prompt).content
    with open(os.path.join(state['output_dir'], "manager_report.md"), "w") as f:
        f.write(report)
        
    return {"manager_report": report}