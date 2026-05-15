import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from config import get_session

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()

MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
session = get_session()
llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1",
    client=session.client("bedrock-runtime"),
)


# ============================================================
# STATE
# ============================================================

class CostAnalysisState(TypedDict):
    messages: Annotated[list, operator.add]
    cost_data: dict
    analysis: str
    alert_needed: bool
    alert_sent: bool
    report: str


# ============================================================
# NODES
# ============================================================

def fetch_costs(state: CostAnalysisState) -> dict:
    """Node 1: Fetch AWS costs"""
    print("📊 Node: fetch_costs")

    cost_data = {
        "ec2": 234.50,
        "rds": 89.20,
        "s3": 12.40,
        "lambda": 4.10,
        "total": 340.20,
        "month": datetime.datetime.now().strftime("%B %Y")
    }

    return {
        "cost_data": cost_data,
        "messages": [AIMessage(content=f"Fetched costs: {cost_data}")]
    }


def analyze_costs(state: CostAnalysisState) -> dict:
    """Node 2: Analyze costs using LLM"""
    print("🧠 Node: analyze_costs")

    cost_data = state["cost_data"]

    response = llm.invoke([
        SystemMessage(content="You are an AWS cost analyst. Be concise."),
        HumanMessage(content=f"""
Analyze these AWS costs and determine if any service exceeds $100:
{cost_data}

Respond in this exact format:
ANALYSIS: <your analysis>
ALERT_NEEDED: <YES or NO>
REASON: <why>
        """)
    ])

    analysis_text = response.content
    alert_needed = "ALERT_NEEDED: YES" in analysis_text.upper()

    print(f"   Analysis complete. Alert needed: {alert_needed}")

    return {
        "analysis": analysis_text,
        "alert_needed": alert_needed,
        "messages": [AIMessage(content=analysis_text)]
    }


def send_alert(state: CostAnalysisState) -> dict:
    """Node 3a: Send Slack alert"""
    print("🔔 Node: send_alert")

    cost_data = state["cost_data"]
    high_cost_services = {
        k: v for k, v in cost_data.items()
        if isinstance(v, float) and v > 100
    }

    alert_msg = (
        f"🚨 AWS Cost Alert — {cost_data['month']}\n"
        f"Services exceeding $100: {high_cost_services}\n"
        f"Total spend: ${cost_data['total']}"
    )

    print(f"   [SLACK] {alert_msg}")

    return {
        "alert_sent": True,
        "messages": [AIMessage(content=f"Alert sent: {alert_msg}")]
    }


def generate_report(state: CostAnalysisState) -> dict:
    """Node 4: Generate final report"""
    print("📝 Node: generate_report")

    cost_data = state["cost_data"]
    alert_sent = state.get("alert_sent", False)

    report = (
        f"AWS Cost Report — {cost_data['month']}\n"
        f"{'='*40}\n"
        f"EC2:     ${cost_data['ec2']:.2f}\n"
        f"RDS:     ${cost_data['rds']:.2f}\n"
        f"S3:      ${cost_data['s3']:.2f}\n"
        f"Lambda:  ${cost_data['lambda']:.2f}\n"
        f"{'='*40}\n"
        f"Total:   ${cost_data['total']:.2f}\n"
        f"Alert sent: {'Yes' if alert_sent else 'No'}"
    )

    print(f"\n{report}")

    return {
        "report": report,
        "messages": [AIMessage(content=report)]
    }


# ============================================================
# ROUTING FUNCTION
# ============================================================

def route_after_analysis(state: CostAnalysisState) -> str:
    """Decide next node based on analysis result"""
    if state["alert_needed"]:
        print("   → Routing to: send_alert")
        return "send_alert"
    else:
        print("   → Routing to: generate_report")
        return "generate_report"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph():
    graph = StateGraph(CostAnalysisState)

    # Add nodes
    graph.add_node("fetch_costs", fetch_costs)
    graph.add_node("analyze_costs", analyze_costs)
    graph.add_node("send_alert", send_alert)
    graph.add_node("generate_report", generate_report)

    # Entry point
    graph.set_entry_point("fetch_costs")

    # Edges
    graph.add_edge("fetch_costs", "analyze_costs")

    # Conditional edge — route based on alert_needed
    graph.add_conditional_edges(
        "analyze_costs",
        route_after_analysis,
        {
            "send_alert": "send_alert",
            "generate_report": "generate_report"
        }
    )

    # After alert → always generate report
    graph.add_edge("send_alert", "generate_report")

    # Report → end
    graph.add_edge("generate_report", END)

    return graph.compile()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app = build_graph()

    # Print graph structure as Mermaid diagram
    print("Graph structure:")
    print(app.get_graph().draw_mermaid())
    print("\n")

    # Run the graph
    print("Running cost analysis graph...\n")

    initial_state = {
        "messages": [HumanMessage(content="Run monthly cost analysis")],
        "cost_data": {},
        "analysis": "",
        "alert_needed": False,
        "alert_sent": False,
        "report": ""
    }

    result = app.invoke(
        initial_state,
        config={"run_name": "cost_analysis_monthly"}
    )

    print("\n" + "="*55)
    print("FINAL STATE:")
    print(f"Alert needed : {result['alert_needed']}")
    print(f"Alert sent   : {result['alert_sent']}")
    print(f"Total messages: {len(result['messages'])}")