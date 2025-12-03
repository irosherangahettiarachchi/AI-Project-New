import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.config import get_llm
from src.state import AgentState

def listing_agent(state: AgentState):
    print("--- [3/7] Listing Agent (LLM) ---")
    llm = get_llm("listing")
    generated_listings = []
    
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
    print("--- [4/7] QA Agent (LLM) ---")
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
            pass 
            
    with open(os.path.join(state['output_dir'], "listing_redlines.json"), "w") as f:
        json.dump(redlines, f, indent=2)

    return {"listing_redlines": redlines}