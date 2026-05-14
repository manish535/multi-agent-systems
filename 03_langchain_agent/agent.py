import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from dotenv import load_dotenv
from config import get_session

# LangChain imports
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# ============================================================
# Step 1 — Initialize ChatBedrock
# ============================================================

session = get_session()

llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1",
    client=session.client("bedrock-runtime"),
)

# ============================================================
# Step 2 — Define tools with @tool decorator
# ============================================================

@tool
def get_aws_cost(service: str) -> str:
    """
    Get the current AWS cost for a specific service.
    Use this when asked about current spending on EC2, RDS, S3, or Lambda.
    Input should be the service name: ec2, rds, s3, or lambda.
    """
    costs = {
        "ec2": "$234.50",
        "rds": "$89.20",
        "s3": "$12.40",
        "lambda": "$4.10"
    }
    return costs.get(service.lower(), f"No data found for: {service}")


@tool
def send_slack_alert(message: str) -> str:
    """
    Send a cost alert message to the #aws-costs Slack channel.
    Use this when costs exceed a threshold or need immediate attention.
    Input should be the alert message string.
    """
    print(f"\n[SLACK] {message}")
    return "Alert delivered to #aws-costs"


@tool
def get_current_month() -> str:
    """
    Get the current month and year.
    Use this when you need to reference the current billing period.
    """
    return datetime.datetime.now().strftime("%B %Y")


@tool
def get_cost_forecast(service: str, months_ahead: int = 1) -> str:
    """
    Forecast future AWS cost for a service based on 10% monthly growth.
    Use this when asked about predicted or future costs.
    Input: service name (ec2/rds/s3/lambda) and months_ahead (default 1).
    """
    current_costs = {
        "ec2": 234.50,
        "rds": 89.20,
        "s3": 12.40,
        "lambda": 4.10
    }
    if service.lower() not in current_costs:
        return f"No forecast data for: {service}"
    current = current_costs[service.lower()]
    forecast = current * (1.10 ** months_ahead)
    return (
        f"Service: {service.upper()} | "
        f"Current: ${current:.2f} | "
        f"Forecast in {months_ahead} month(s): ${forecast:.2f} | "
        f"Growth rate: 10%/month"
    )


tools = [get_aws_cost, send_slack_alert, get_current_month, get_cost_forecast]

# ============================================================
# Step 3 — Create agent
# ============================================================
memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are an AWS cost analyst. "
        "Be concise and precise. "
        "Always use tools to get real data before answering."
    ),
    checkpointer=memory
)

# ============================================================
# Step 4 — Run tasks
# ============================================================

def run_task(task: str, thread_id: str = "default"):
    print(f"\n{'='*55}")
    print(f"TASK: {task}")
    print(f"{'='*55}")

    # thread_id ties messages together for memory tracking
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config=config
    )

    # LangChain returns messages list — last one is final answer
    final = result["messages"][-1].content
    if isinstance(final, list):
        final = final[0].get("text", str(final))
    print(f"\n Final Answer: {final}")

    # Token usage — LangChain tracks this automatically
    last_msg = result["messages"][-1]
    if hasattr(last_msg, "usage_metadata") and last_msg.usage_metadata:
        usage = last_msg.usage_metadata
        print(f"\n Token Usage:")
        print(f"   Input tokens : {usage.get('input_tokens', 'N/A')}")
        print(f"   Output tokens: {usage.get('output_tokens', 'N/A')}")

    return final


if __name__ == "__main__":
    # Task 1 — simple
    run_task("What is the EC2 cost this month?", thread_id="task1")

    print("\n\n")

    # Task 2 — multi-step conditional
    run_task("Check EC2 and RDS costs. If either exceeds $100, send a Slack alert.")

    print("\n\n")

    # Task 3 — forecast
    run_task("What will EC2 cost in 3 months?", thread_id="forecast-task")

    # Memory problem demonstration
    print("\n\n=== MEMORY PROBLEM DEMO ===")
    run_task("My name is Manish and I am monitoring AWS costs", thread_id="memory-demo")
    run_task("What is my name?", thread_id="memory-demo")