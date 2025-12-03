from langgraph.graph import StateGraph, END
from src.state import AgentState

# Import agents from their respective modules
from src.agents.inventory import sourcing_agent, pricing_agent
from src.agents.content import listing_agent, qa_agent
from src.agents.ops import order_routing_agent, reporter_agent, manager_agent

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("sourcing", sourcing_agent)
    workflow.add_node("pricing", pricing_agent)
    workflow.add_node("listing", listing_agent)
    workflow.add_node("qa", qa_agent)
    workflow.add_node("routing", order_routing_agent)
    workflow.add_node("reporting", reporter_agent)
    workflow.add_node("manager", manager_agent)
    
    # Define Edges
    workflow.set_entry_point("sourcing")
    workflow.add_edge("sourcing", "pricing")
    workflow.add_edge("pricing", "listing")
    workflow.add_edge("listing", "qa")
    workflow.add_edge("qa", "routing")
    workflow.add_edge("routing", "reporting")
    workflow.add_edge("reporting", "manager")
    workflow.add_edge("manager", END)
    
    return workflow.compile()