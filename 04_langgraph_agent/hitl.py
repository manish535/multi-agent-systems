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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
session = get_session()
llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1",
    client=session.client("bedrock-runtime"),
)


# ============================================================
# STATE — same as Day 9 + human_decision field
# ============================================================

class CostAnalysisState(TypedDict):
    messages: Annotated[list, operator.add]
    cost_data: dict
    analysis: str
    alert_needed: bool
    alert_sent: bool
    human_decision: str   # "approved" | "rejected" | ""
    report: str


# ============================================================
# NODES
# ============================================================

def fetch_costs(state: CostAnalysisState) -> dict:
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


def human_review(state: CostAnalysisState) -> dict:
    """
    HITL Node — graph PAUSES here.
    Shows the human what the agent found and asks for approval.
    The interrupt() call freezes execution and saves state to checkpointer.
    When resumed, the value passed to Command(resume=...) becomes
    the return value of interrupt().
    """
    print("\n" + "="*55)
    print("⏸  PAUSED — Human approval required")
    print("="*55)

    cost_data = state["cost_data"]
    analysis = state["analysis"]

    # Show human what agent found
    print(f"\nAgent analysis:\n{analysis}")
    print(f"\nHigh cost services:")
    for k, v in cost_data.items():
        if isinstance(v, float) and v > 100:
            print(f"  {k.upper()}: ${v:.2f}")

    # THIS IS WHERE THE GRAPH PAUSES
    # interrupt() saves state and waits for human input
    # The string passed to interrupt() is shown to the human
    decision = interrupt(
        f"Alert needed for high AWS costs. "
        f"EC2: ${cost_data['ec2']}, Total: ${cost_data['total']}. "
        f"Approve sending Slack alert? (approved/rejected)"
    )

    # When resumed, decision = whatever was passed to Command(resume=...)
    print(f"\n✅ Human decision received: {decision}")

    return {
        "human_decision": decision,
        "messages": [AIMessage(content=f"Human decision: {decision}")]
    }


def send_alert(state: CostAnalysisState) -> dict:
    print("🔔 Node: send_alert")

    # Only send if human approved
    if state["human_decision"] != "approved":
        print("   ❌ Alert rejected by human — skipping")
        return {
            "alert_sent": False,
            "messages": [AIMessage(content="Alert rejected by human")]
        }

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
    print("📝 Node: generate_report")
    cost_data = state["cost_data"]
    alert_sent = state.get("alert_sent", False)
    human_decision = state.get("human_decision", "N/A")

    report = (
        f"AWS Cost Report — {cost_data['month']}\n"
        f"{'='*40}\n"
        f"EC2:     ${cost_data['ec2']:.2f}\n"
        f"RDS:     ${cost_data['rds']:.2f}\n"
        f"S3:      ${cost_data['s3']:.2f}\n"
        f"Lambda:  ${cost_data['lambda']:.2f}\n"
        f"{'='*40}\n"
        f"Total:   ${cost_data['total']:.2f}\n"
        f"Human decision : {human_decision}\n"
        f"Alert sent     : {'Yes' if alert_sent else 'No'}"
    )
    print(f"\n{report}")
    return {
        "report": report,
        "messages": [AIMessage(content=report)]
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_analysis(state: CostAnalysisState) -> str:
    if state["alert_needed"]:
        print("   → Routing to: human_review (HITL)")
        return "human_review"
    else:
        print("   → Routing to: generate_report (no alert needed)")
        return "generate_report"


# ============================================================
# BUILD GRAPH — with checkpointer for HITL
# ============================================================

def build_graph():
    graph = StateGraph(CostAnalysisState)

    graph.add_node("fetch_costs", fetch_costs)
    graph.add_node("analyze_costs", analyze_costs)
    graph.add_node("human_review", human_review)   # NEW
    graph.add_node("send_alert", send_alert)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("fetch_costs")
    graph.add_edge("fetch_costs", "analyze_costs")

    # After analysis — route to human_review if alert needed
    graph.add_conditional_edges(
        "analyze_costs",
        route_after_analysis,
        {
            "human_review": "human_review",
            "generate_report": "generate_report"
        }
    )

    graph.add_edge("human_review", "send_alert")
    graph.add_edge("send_alert", "generate_report")
    graph.add_edge("generate_report", END)

    # CHECKPOINTER — required for HITL
    # Saves state so graph can be resumed after pause
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ============================================================
# RUN — two-phase execution
# ============================================================

if __name__ == "__main__":
    app = build_graph()

    # thread_id ties the two invocations together
    # same thread = same conversation = graph can resume
    config = {"configurable": {"thread_id": "cost-analysis-001"}}

    initial_state = {
        "messages": [HumanMessage(content="Run monthly cost analysis")],
        "cost_data": {},
        "analysis": "",
        "alert_needed": False,
        "alert_sent": False,
        "human_decision": "",
        "report": ""
    }

    print("="*55)
    print("PHASE 1: Running graph until HITL pause...")
    print("="*55)

    # PHASE 1 — Run until interrupt
    result = app.invoke(
        initial_state, 
        config = {
            "configurable": {"thread_id": "cost-analysis-001"},
            "run_name": "cost_analysis_hitl",
            "tags": ["cost-analysis", "hitl"],
            "metadata": {"version": "1.0", "env": "dev"}
        }
    )

    print("\n" + "="*55)
    print("Graph is PAUSED. State saved to checkpointer.")
    print("Waiting for human decision...")
    print("="*55)

    # Simulate human reviewing and deciding
    # In production this would be a Slack button, web UI, API call
    print("\n👤 Human reviewing the alert...")
    print("   (In production: Slack button / web UI / API)")

    human_input = input("\n   Your decision (approved/rejected): ").strip().lower()

    print("\n" + "="*55)
    print("PHASE 2: Resuming graph with human decision...")
    print("="*55)

    # PHASE 2 — Resume with human decision
    # Command(resume=...) passes the value back to interrupt()
    result = app.invoke(
        Command(resume=human_input),
        config = {
            "configurable": {"thread_id": "cost-analysis-001"},
            "run_name": "cost_analysis_hitl",
            "tags": ["cost-analysis", "hitl"],
            "metadata": {"version": "1.0", "env": "dev"}
        }
    )

    print("\n" + "="*55)
    print("FINAL STATE:")
    print(f"Human decision : {result['human_decision']}")
    print(f"Alert sent     : {result['alert_sent']}")
    print(f"Total messages : {len(result['messages'])}")
