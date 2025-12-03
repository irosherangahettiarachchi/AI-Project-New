import os
import json
import math
import pandas as pd
from src.state import AgentState

def sourcing_agent(state: AgentState):
    print("--- [1/7] Product Sourcing Agent ---")
    df = pd.read_csv(state['catalog_path'])
    
    # Criteria: Stock >= 10. 
    filtered = df[df['stock'] >= 10].copy()
    
    # Pick top 10 based on stock level
    top_10 = filtered.sort_values(by='stock', ascending=False).head(10)
    
    selected = top_10.to_dict(orient='records')
    
    # Save artifact
    with open(os.path.join(state['output_dir'], "selection.json"), "w") as f:
        json.dump(selected, f, indent=2)
        
    return {"selected_skus": selected, "raw_catalog": df.to_dict(orient='records')}

def pricing_agent(state: AgentState):
    print("--- [2/7] Pricing & Stock Agent ---")
    selected = state['selected_skus']
    price_updates = []
    stock_updates = []
    
    # Pricing Formula: P = (Cost + Shipping + 0.30) / 0.621
    divisor = 0.621 
    
    for item in selected:
        cost = float(item['cost_price'])
        shipping = float(item['shipping_cost'])
        
        min_price = (cost + shipping + 0.30) / divisor
        final_price = math.ceil(min_price * 2) / 2 # Round up to nearest 0.50
        
        price_updates.append({
            "sku": item['supplier_sku'],
            "new_price": final_price,
            "cost_basis": cost + shipping
        })
        
        stock_updates.append({
            "sku": item['supplier_sku'],
            "stock_level": item['stock']
        })

    pd.DataFrame(price_updates).to_csv(os.path.join(state['output_dir'], "price_update.csv"), index=False)
    pd.DataFrame(stock_updates).to_csv(os.path.join(state['output_dir'], "stock_update.csv"), index=False)
    
    return {"price_updates": price_updates, "stock_updates": stock_updates}