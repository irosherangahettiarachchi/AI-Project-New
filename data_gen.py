import pandas as pd
import random
import os

def generate_data():
    os.makedirs("data", exist_ok=True)
    
    # 1. Generate Supplier Catalog (30 SKUs)
    categories = ["Electronics", "Home", "Fitness", "Accessories"]
    catalog_data = []
    
    for i in range(1, 31):
        cost = round(random.uniform(5.0, 50.0), 2)
        stock = random.randint(0, 50) # Some will be < 10 to test filtering
        catalog_data.append({
            "supplier_sku": f"SKU-{1000+i}",
            "name": f"Generic Product {i}",
            "category": random.choice(categories),
            "cost_price": cost,
            "stock": stock,
            "weight_kg": round(random.uniform(0.1, 2.0), 1),
            "length_cm": 10, "width_cm": 10, "height_cm": 10,
            "image_url": f"http://img.com/{i}.jpg",
            "description": f"A high quality generic product {i} for your needs.",
            "brand": "GenericBrand",
            "shipping_cost": 5.00,
            "supplier_lead_days": 3
        })
    
    pd.DataFrame(catalog_data).to_csv("data/supplier_catalog.csv", index=False)
    print("Generated data/supplier_catalog.csv")

    # 2. Generate Orders
    orders_data = []
    for i in range(1, 6):
        orders_data.append({
            "order_id": f"ORD-{5000+i}",
            "sku": f"SKU-{1000+random.randint(1,30)}", # Random SKU
            "quantity": random.randint(1, 2),
            "customer_country": random.choice(["US", "AU", "UK"]),
            "order_date": "2023-10-27"
        })
        
    pd.DataFrame(orders_data).to_csv("data/orders.csv", index=False)
    print("Generated data/orders.csv")

if __name__ == "__main__":
    generate_data()