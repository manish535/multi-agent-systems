# Same graph as graph.py — but every node tracked in MLflow

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

# Import MLflow tracker
from observability import NodeTracker, AgentRunTracker

load_dotenv()

MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
session  = get_session()
llm      = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1",
    client=session.client("bedrock-runtime"),
)


# ============================================================
# STATE
# ============================================================

class CostAnalysisState(TypedDict):
    messages:     Annotated[list, operator.add]
    cost_data:    dict
    analysis:     str
    alert_needed: bool
    alert_sent:   bool
    report:       str


# ============================================================
# NODES — same logic as graph.py + MLflow tracking
# ============================================================

def fetch_costs(state: CostAnalysisState) -> dict:
    print("📊 Node: fetch_costs")

    with NodeTracker("cost_agent", "fetch_costs"):
        cost_data = {
            "ec2":    234.50,
            "rds":    89.20,
            "s3":     12.40,
            "lambda": 4.10,
            "total":  340.20,
            "month":  datetime.datetime.now().strftime("%B %Y")
        }

    return {
        "cost_data": cost_data,
        "messages":  [AIMessage(content=f"Fetched costs: {cost_data}")]
    }


def analyze_costs(state: CostAnalysisState) -> dict:
    print("🧠 Node: analyze_costs")

    with NodeTracker("cost_agent", "analyze_costs") as tracker:
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
        alert_needed  = "ALERT_NEEDED: YES" in analysis_text.upper()

        # Track LLM token usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tracker.log_tokens(
                response.usage_metadata.get("input_tokens", 0),
                response.usage_metadata.get("output_tokens", 0)
            )

        # Track routing decision
        tracker.log_routing("send_alert" if alert_needed else "generate_report")

        print(f"   Alert needed: {alert_needed}")

    return {
        "analysis":     analysis_text,
        "alert_needed": alert_needed,
        "messages":     [AIMessage(content=analysis_text)]
    }


def send_alert(state: CostAnalysisState) -> dict:
    print("🔔 Node: send_alert")

    with NodeTracker("cost_agent", "send_alert"):
        cost_data = state["cost_data"]
        high_cost = {
            k: v for k, v in cost_data.items()
            if isinstance(v, float) and v > 100
        }
        alert_msg = (
            f"🚨 AWS Cost Alert — {cost_data['month']}\n"
            f"Services > $100: {high_cost}\n"
            f"Total: ${cost_data['total']}"
        )
        print(f"   [SLACK] {alert_msg}")

    return {
        "alert_sent": True,
        "messages":   [AIMessage(content=f"Alert sent: {alert_msg}")]
    }


def generate_report(state: CostAnalysisState) -> dict:
    print("📝 Node: generate_report")

    with NodeTracker("cost_agent", "generate_report"):
        cost_data  = state["cost_data"]
        alert_sent = state.get("alert_sent", False)

        report = (
            f"AWS Cost Report — {cost_data['month']}\n"
            f"{'='*40}\n"
            f"EC2:    ${cost_data['ec2']:.2f}\n"
            f"RDS:    ${cost_data['rds']:.2f}\n"
            f"S3:     ${cost_data['s3']:.2f}\n"
            f"Lambda: ${cost_data['lambda']:.2f}\n"
            f"{'='*40}\n"
            f"Total:  ${cost_data['total']:.2f}\n"
            f"Alert:  {'Sent ✅' if alert_sent else 'Not needed'}"
        )
        print(f"\n{report}")

    return {
        "report":   report,
        "messages": [AIMessage(content=report)]
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_analysis(state: CostAnalysisState) -> str:
    if state["alert_needed"]:
        return "send_alert"
    return "generate_report"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():
    graph = StateGraph(CostAnalysisState)

    graph.add_node("fetch_costs",     fetch_costs)
    graph.add_node("analyze_costs",   analyze_costs)
    graph.add_node("send_alert",      send_alert)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("fetch_costs")
    graph.add_edge("fetch_costs", "analyze_costs")

    graph.add_conditional_edges(
        "analyze_costs",
        route_after_analysis,
        {
            "send_alert":      "send_alert",
            "generate_report": "generate_report"
        }
    )

    graph.add_edge("send_alert",      "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ============================================================
# RUN WITH MLFLOW TRACKING
# ============================================================

if __name__ == "__main__":
    app = build_graph()

    print("Graph structure:")
    print(app.get_graph().draw_mermaid())
    print()

    # ── Run 1: Standard cost analysis ──
    with AgentRunTracker(
        "cost_analysis",
        channel="#aws-costs",
        intent="COST_QUERY"
    ) as run:

        initial_state = {
            "messages":     [HumanMessage(content="Run monthly cost analysis")],
            "cost_data":    {},
            "analysis":     "",
            "alert_needed": False,
            "alert_sent":   False,
            "report":       ""
        }

        result = app.invoke(initial_state)

        run.log_outcome(
            "success",
            alert_sent=int(result["alert_sent"]),
            total_messages=len(result["messages"])
        )
        run.log_state(result)

    print("\n✅ Run 1 complete — check MLflow UI\n")
    print("="*55)

    # ── Run 2: Simulate no alert needed ──
    # Temporarily patch costs below threshold
    def fetch_low_costs(state):
        return {
            "cost_data": {
                "ec2": 45.00, "rds": 30.00,
                "s3": 5.00, "lambda": 2.00,
                "total": 82.00,
                "month": datetime.datetime.now().strftime("%B %Y")
            },
            "messages": [AIMessage(content="Fetched low costs")]
        }

    # Build second graph with low costs
    graph2 = StateGraph(CostAnalysisState)
    graph2.add_node("fetch_costs",     fetch_low_costs)
    graph2.add_node("analyze_costs",   analyze_costs)
    graph2.add_node("send_alert",      send_alert)
    graph2.add_node("generate_report", generate_report)
    graph2.set_entry_point("fetch_costs")
    graph2.add_edge("fetch_costs", "analyze_costs")
    graph2.add_conditional_edges(
        "analyze_costs", route_after_analysis,
        {"send_alert": "send_alert", "generate_report": "generate_report"}
    )
    graph2.add_edge("send_alert", "generate_report")
    graph2.add_edge("generate_report", END)
    app2 = graph2.compile()

    with AgentRunTracker(
        "cost_analysis",
        channel="#aws-costs",
        intent="COST_QUERY"
    ) as run:

        result2 = app2.invoke({
            "messages":     [HumanMessage(content="Run cost analysis")],
            "cost_data":    {},
            "analysis":     "",
            "alert_needed": False,
            "alert_sent":   False,
            "report":       ""
        })

        run.log_outcome(
            "success",
            alert_sent=int(result2["alert_sent"]),
            total_messages=len(result2["messages"])
        )
        run.log_state(result2)

    print("\n✅ Run 2 complete — check MLflow UI\n")
    print(f"\n🎯 Open MLflow: http://localhost:5001")
    print(f"   Experiment: multi-agent-oncall")
    print(f"   You should see 2 runs with different outcomes")