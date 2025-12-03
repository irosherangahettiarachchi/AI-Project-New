from typing import List, Dict, TypedDict

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