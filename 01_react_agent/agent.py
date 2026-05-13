import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
from typing import Callable
from config import get_bedrock_client
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

# ============================================================
# ReActAgent Class
# ============================================================

class ReActAgent:
    """
    A reusable ReAct agent that works with AWS Bedrock.    
    """

    def __init__(
        self,
        tools: list,
        model_id: str = MODEL_ID,
        max_iterations: int = 10,
        system_prompt: str = None,
        verbose: bool = True
    ):
        self.client = get_bedrock_client()
        self.model_id = model_id
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.system_prompt = system_prompt or "You are a helpful AI assistant."

        # Build tool registry + specs from tool list
        self.tool_functions = {}
        self.tool_specs = []
        for tool in tools:
            self._register_tool(tool)

        # Token tracking — important for cost control in production
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _register_tool(self, tool: dict):
        """
        Register a tool with the agent.
        Each tool is a dict with: name, description, function, parameters
        """
        self.tool_functions[tool["name"]] = tool["function"]
        self.tool_specs.append({
            "toolSpec": {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": tool["parameters"],
                        "required": tool.get("required", [])
                    }
                }
            }
        })

    def _log(self, message: str):
        """Print only if verbose mode is on"""
        if self.verbose:
            print(message)

    def _call_llm(self, messages: list) -> dict:
        """
        Call Bedrock with current messages.
        Tracks token usage on every call.
        """
        response = self.client.converse(
            modelId=self.model_id,
            system=[{"text": self.system_prompt}],
            messages=messages,
            toolConfig={"tools": self.tool_specs}
        )

        # Track tokens — every call costs money
        usage = response.get("usage", {})
        self.total_input_tokens += usage.get("inputTokens", 0)
        self.total_output_tokens += usage.get("outputTokens", 0)

        return response

    def _run_tool(self, tool_name: str, tool_input: dict) -> str:
        """
        Run a tool safely with error handling.
        Returns error message instead of crashing if tool fails.
        """
        if tool_name not in self.tool_functions:
            return f"Error: tool '{tool_name}' not found"

        try:
            result = self.tool_functions[tool_name](**tool_input)
            return str(result)
        except Exception as e:
            # Don't crash the agent — return error as observation
            # Claude will see this and decide what to do next
            return f"Tool error: {str(e)}"

    def run(self, task: str) -> str:
        """
        Run the agent on a task.
        Returns the final answer as a string.
        """
        self._log(f"\n{'='*55}")
        self._log(f"TASK: {task}")
        self._log(f"{'='*55}\n")

        # Reset token counter for this run
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        messages = [
            {"role": "user", "content": [{"text": task}]}
        ]

        final_answer = None

        for iteration in range(1, self.max_iterations + 1):
            self._log(f"--- Iteration {iteration} ---")

            try:
                response = self._call_llm(messages)
            except Exception as e:
                self._log(f" LLM call failed: {e}")
                return f"Agent failed: {str(e)}"

            stop_reason = response["stopReason"]
            response_message = response["output"]["message"]
            messages.append(response_message)

            # Claude is done
            if stop_reason == "end_turn":
                final_answer = response_message["content"][0]["text"]
                self._log(f"\n Final Answer: {final_answer}")
                break

            # Claude wants to use tools
            if stop_reason == "tool_use":
                tool_results = []

                for block in response_message["content"]:
                    if "text" in block and block["text"].strip():
                        self._log(f"Thought: {block['text']}")

                    if "toolUse" in block:
                        tool_name = block["toolUse"]["name"]
                        tool_input = block["toolUse"]["input"]
                        tool_use_id = block["toolUse"]["toolUseId"]

                        self._log(f"Action: {tool_name}({tool_input})")

                        # Safe tool execution
                        result = self._run_tool(tool_name, tool_input)
                        self._log(f"Observation: {result}\n")

                        tool_results.append({
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": result}]
                            }
                        })

                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

        # Print token usage summary
        self._log(f"\n Token Usage:")
        self._log(f"   Input tokens : {self.total_input_tokens}")
        self._log(f"   Output tokens: {self.total_output_tokens}")
        self._log(f"   Total tokens : {self.total_input_tokens + self.total_output_tokens}")
        self._log(f"   Est. cost    : ${((self.total_input_tokens * 0.0008) + (self.total_output_tokens * 0.0024)) / 1000:.6f}")

        return final_answer or "Agent reached max iterations without final answer"


# ============================================================
# Tool Definitions — clean format for ReActAgent
# ============================================================

def get_aws_cost(service: str) -> str:
    costs = {
        "ec2": "$234.50",
        "rds": "$89.20",
        "s3": "$12.40",
        "lambda": "$4.10"
    }
    return costs.get(service.lower(), f"No data found for: {service}")


def send_slack_alert(message: str) -> str:
    print(f"\n[SLACK] {message}")
    return "Alert delivered to #aws-costs"


def get_current_month() -> str:
    return datetime.datetime.now().strftime("%B %Y")

def get_cost_forecast(service: str, months_ahead: int = 1) -> str:
    """Predict future AWS cost based on current spend + 10% monthly growth"""
    current_costs = {
        "ec2": 234.50,
        "rds": 89.20,
        "s3": 12.40,
        "lambda": 4.10
    }

    if service.lower() not in current_costs:
        return f"No forecast data available for: {service}"

    current = current_costs[service.lower()]
    forecast = current * ((1.10) ** months_ahead)

    return (
        f"Service: {service.upper()} | "
        f"Current: ${current:.2f} | "
        f"Forecast in {months_ahead} month(s): ${forecast:.2f} | "
        f"Growth rate: 10%/month"
    )

# Tool definitions — notice the clean dict format
# name, description, function, parameters, required
aws_tools = [
    {
        "name": "get_aws_cost",
        "description": "Get AWS cost for a specific service like EC2, RDS, S3, Lambda",
        "function": get_aws_cost,
        "parameters": {
            "service": {
                "type": "string",
                "description": "AWS service name e.g. ec2, rds, s3, lambda"
            }
        },
        "required": ["service"]
    },
    {
        "name": "get_cost_forecast",
        "description": "Predict future AWS cost for a service based on current spend and growth trend",
        "function": get_cost_forecast,
        "parameters": {
            "service": {
                "type": "string",
                "description": "AWS service name e.g. ec2, rds, s3, lambda"
            },
            "months_ahead": {
                "type": "integer",
                "description": "Number of months ahead to forecast. Default is 1."
            }
        },
        "required": ["service"]
    },
    {
        "name": "send_slack_alert",
        "description": "Send a cost alert message to the Slack channel",
        "function": send_slack_alert,
        "parameters": {
            "message": {
                "type": "string",
                "description": "The alert message to send"
            }
        },
        "required": ["message"]
    },
    {
        "name": "get_current_month",
        "description": "Get the current month and year",
        "function": get_current_month,
        "parameters": {},
        "required": []
    }
]


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    # Create agent instance
    agent = ReActAgent(
        tools=aws_tools,
        system_prompt="You are an AWS cost analyst. Be concise and precise.",
        verbose=True
    )

    # Task 1 — simple
    agent.run("What is the EC2 cost this month?")

    print("\n\n")

    # Task 2 — multi-step conditional
    agent.run("Check EC2 and RDS costs. If either exceeds $100, send a Slack alert.")

    print("\n\n")

    # Task 3 — error handling test (tool doesn't exist)
    agent.run("What is the EKS cost this month?")

    # Task 4 — forecast
    agent.run("What will EC2 cost in 3 months?")

    print("\n\n")

    # Task 5 — combined analysis
    agent.run(
        "Check current EC2 and RDS costs. "
        "Forecast both for next 2 months. "
        "If forecast exceeds $300 for any service, send a Slack alert."
    )