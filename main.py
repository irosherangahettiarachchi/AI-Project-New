import os
import argparse
from src.graph import build_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shopify Dropshipping Ops Agent")
    parser.add_argument("--catalog", required=True, help="Path to supplier CSV")
    parser.add_argument("--orders", required=True, help="Path to orders CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Ensure output dir exists
    os.makedirs(args.out, exist_ok=True)
    
    print(f"Starting Ops Agent...\nInputs: {args.catalog}, {args.orders}\nOutput: {args.out}")
    
    # Initialize State
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
    
    # Build and Run
    app = build_graph()
    app.invoke(initial_state)
    
    print("\nWorkflow Complete. Check 'out/' directory.")