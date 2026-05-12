import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
from config import get_bedrock_client
from dotenv import load_dotenv

load_dotenv()

client = get_bedrock_client()
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")


def get_aws_cost(service: str) -> str:
    costs = {
        "ec2": "$234.50",
        "rds": "$89.20",
        "s3": "$12.40",
        "lambda": "$4.10"
    }
    return costs.get(service.lower(), f"No data found for: {service}")


def send_slack_alert(message: str) -> str:
    print(f"\n[SLACK] 🔔 {message}")
    return "Alert delivered to #aws-costs"


def get_current_month() -> str:
    return datetime.datetime.now().strftime("%B %Y")


tool_functions = {
    "get_aws_cost": get_aws_cost,
    "send_slack_alert": send_slack_alert,
    "get_current_month": get_current_month
}

tool_specs = [
    {
        "toolSpec": {
            "name": "get_aws_cost",
            "description": "Get AWS cost for a specific service like EC2, RDS, S3, Lambda",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "AWS service name e.g. ec2, rds, s3, lambda"
                        }
                    },
                    "required": ["service"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "send_slack_alert",
            "description": "Send a cost alert message to the Slack channel",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The alert message to send"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "get_current_month",
            "description": "Get the current month and year",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    }
]


def run_agent(user_task: str):
    print(f"\n{'='*55}")
    print(f"TASK: {user_task}")
    print(f"{'='*55}\n")

    messages = [
        {"role": "user", "content": [{"text": user_task}]}
    ]

    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        iteration += 1
        print(f"--- Iteration {iteration} ---")

        response = client.converse(
            modelId=MODEL_ID,
            messages=messages,
            toolConfig={"tools": tool_specs}
        )

        stop_reason = response["stopReason"]
        response_message = response["output"]["message"]
        messages.append(response_message)

        # Claude finished
        if stop_reason == "end_turn":
            final = response_message["content"][0]["text"]
            print(f"\n✅ Final Answer: {final}")
            break

        # Claude wants to use a tool
        if stop_reason == "tool_use":
            tool_results = []

            for block in response_message["content"]:
                if "text" in block and block["text"].strip():
                    print(f"Thought: {block['text']}")

                if "toolUse" in block:
                    tool_name = block["toolUse"]["name"]
                    tool_input = block["toolUse"]["input"]
                    tool_use_id = block["toolUse"]["toolUseId"]

                    print(f"Action: {tool_name}({tool_input})")

                    fn = tool_functions[tool_name]
                    result = fn(**tool_input)

                    print(f"Observation: {result}\n")

                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"text": str(result)}]
                        }
                    })

            if tool_results:
                messages.append({
                    "role": "user",
                    "content": tool_results
                })


if __name__ == "__main__":
    run_agent("What is the EC2 cost this month?")
    print("\n\n")
    run_agent("Check EC2 and RDS costs. If either exceeds $100, send a Slack alert.")