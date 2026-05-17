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
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# MLflow tracking
import mlflow
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../04_langgraph_agent")
from observability import AgentRunTracker, NodeTracker

load_dotenv()

MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
session  = get_session()

llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1",
    client=session.client("bedrock-runtime"),
)

# ============================================================
# STATE
# ============================================================

class OncallAgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ============================================================
# TOOLS
# Real tools your oncall agent needs
# Replace mock returns with real API calls in production
# ============================================================

@tool
def get_aws_cost(service: str) -> str:
    """
    Get current AWS cost for a service.
    Input: service name (ec2, rds, s3, lambda)
    Use when asked about AWS costs or spending.
    """
    costs = {
        "ec2":    "$234.50",
        "rds":    "$89.20",
        "s3":     "$12.40",
        "lambda": "$4.10"
    }
    return costs.get(service.lower(), f"No cost data for: {service}")


@tool
def check_github_pr(pr_url: str) -> str:
    """
    Check GitHub PR status and validate against PAR Tech merge requirements.
    Input: GitHub PR URL from punchh/script-tasks repo.
    """
    import urllib.request
    import json
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from rag import search_knowledge_base

    # Get GitHub token from Secrets Manager
    sm    = get_session().client("secretsmanager", region_name="us-east-1")
    token = sm.get_secret_value(
        SecretId=os.getenv("GITHUB_TOKEN_SECRET", "cost-optimizer/github-token")
    )["SecretString"]

    # Parse PR URL → owner/repo/number
    parts     = pr_url.rstrip("/").split("/")
    owner     = parts[-4]
    repo      = parts[-3]
    pr_number = parts[-1]

    # Call GitHub API
    req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}",
        headers={
            "Authorization": f"token {token}",
            "Accept":        "application/vnd.github+json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            pr = json.loads(resp.read())
    except Exception as e:
        return f"GitHub API failed: {str(e)}"

    # Get PAR Tech merge rules from S3
    merge_rules = search_knowledge_base(
        "PR merge checklist DBA approval required fields"
    )

    return {
        "pr_number":   pr_number,
        "title":       pr.get("title", ""),
        "status":      pr.get("state", ""),
        "author":      pr.get("user", {}).get("login", ""),
        "description": pr.get("body", ""),
        "mergeable":   pr.get("mergeable", False),
        "merge_rules": merge_rules
    }

@tool
def check_k8s_pods(namespace: str, service: str = "") -> str:
    """
    Check Kubernetes pod status in a namespace.
    Input: namespace name, optional service name
    Use when there are environment issues or pods crashing.
    """
    # In production: call K8s API via kubectl or python-kubernetes
    return {
        "namespace": namespace,
        "service":   service,
        "pods": [
            {"name": f"{service}-7d4b9c-xk2p", "status": "Running",   "restarts": 0},
            {"name": f"{service}-7d4b9c-mn8q", "status": "OOMKilled", "restarts": 5},
        ],
        "issue": "Pod OOMKilled — memory limit exceeded",
        "recommended_action": "Increase memory limit or investigate memory leak"
    }


@tool
def send_slack_message(channel: str, message: str) -> str:
    """
    Send a message to a Slack channel.
    Input: channel name and message text.
    Use to notify team or respond to oncall requests.
    """
    print(f"\n[SLACK → {channel}] {message}")
    return f"Message sent to {channel}"


@tool
def get_current_month() -> str:
    """Get current month and year."""
    return datetime.datetime.now().strftime("%B %Y")


tools     = [get_aws_cost, check_github_pr, check_k8s_pods,
             send_slack_message, get_current_month]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# NODES
# ============================================================

def call_llm(state: OncallAgentState) -> dict:
    """Call LLM — decides tool call or final answer."""
    print("🧠 Node: call_llm")

    with NodeTracker("oncall_agent", "call_llm") as tracker:
        system = SystemMessage(content=(
            "You are an oncall DevOps agent for PAR Technology (Punchh platform). "
            "You help with AWS costs, GitHub PR merges, Kubernetes issues, and alerts. "
            "Always use tools to get real data. Be concise and actionable.\n\n"
            "For PR merge requests: "
            "1. Call check_github_pr to get PR data and merge rules from Confluence runbooks. "
            "2. Validate EVERY required field is present in PR description. "
            "3. Check if DBA approval is needed based on the rules. "
            "4. Give clear YES/NO merge decision with specific reasons. "
            "5. If DBA approval needed — say exactly that and stop."
        ))

        response = llm_with_tools.invoke([system] + state["messages"])

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tracker.log_tokens(
                response.usage_metadata.get("input_tokens", 0),
                response.usage_metadata.get("output_tokens", 0)
            )

        decision = "tool_call" if response.tool_calls else "final_answer"
        tracker.log_routing(decision)
        print(f"   Decision: {decision}")

    return {"messages": [response]}


# ============================================================
# ROUTING
# ============================================================

def should_continue(state: OncallAgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        print(f"   → tools ({[tc['name'] for tc in last.tool_calls]})")
        return "tools"
    print("   → END")
    return "end"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():
    graph = StateGraph(OncallAgentState)

    graph.add_node("call_llm", call_llm)
    graph.add_node("tools",    tool_node)

    graph.set_entry_point("call_llm")

    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {"tools": "tools", "end": END}
    )

    graph.add_edge("tools", "call_llm")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


agent_graph = build_graph()

# ============================================================
# AGENTCORE WRAPPER
# ============================================================

agentcore_app = BedrockAgentCoreApp()

@agentcore_app.entrypoint
def agent_handler(payload, context):
    """
    AgentCore calls this for every request.
    payload: {"message": "...", "session_id": "...", "channel": "..."}
    """
    message    = payload.get("message", "")
    session_id = payload.get("session_id", "default")
    channel    = payload.get("channel", "unknown")

    print(f"\n{'='*55}")
    print(f"Channel:  {channel}")
    print(f"Message:  {message}")
    print(f"Session:  {session_id}")
    print(f"{'='*55}")

    with AgentRunTracker("oncall_agent", channel=channel) as run:
        config = {"configurable": {"thread_id": session_id}}

        result = agent_graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        final = result["messages"][-1].content
        if isinstance(final, list):
            final = final[0].get("text", str(final))

        last_msg = result["messages"][-1]
        input_tokens  = 0
        output_tokens = 0
        if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
            input_tokens  = last_msg.usage_metadata.get("input_tokens", 0)
            output_tokens = last_msg.usage_metadata.get("output_tokens", 0)

        run.log_outcome(
            "success",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_messages=len(result["messages"])
        )

    return {
        "response":   final,
        "session_id": session_id,
        "channel":    channel,
        "usage": {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens
        }
    }


# ============================================================
# LOCAL TEST — simulate your real oncall scenarios
# ============================================================

if __name__ == "__main__":
    print("Testing oncall agent locally...\n")
    print("Graph:")
    print(agent_graph.get_graph().draw_mermaid())
    print()

    # Simulate real @oncall-punchh-devops scenarios
    test_cases = [
        # {
        #     "message":    "What is EC2 cost this month?",
        #     "session_id": "test-cost-001",
        #     "channel":    "#aws-costs"
        # },
        {
            "message":    "Please merge this PR: https://github.com/punchh/script-tasks/pull/2301",    
            "session_id": "test-pr-001",
            "channel":    "#git-script-repo"
        }#,
        # {
        #     "message":    "developer instance not getting up, pods crashing in dev namespace",
        #     "session_id": "test-env-001",
        #     "channel":    "#devops-non-prod-punchh"
        # }
    ]


    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*55}")
        print(f"Test {i} — {test['channel']}")
        print(f"{'='*55}")

        result = agent_handler(test, context={})

        print(f"\n✅ Response: {result['response']}")
        print(f"   Tokens:   {result['usage']}")
        print(f"   Session:  {result['session_id']}")