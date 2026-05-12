def get_aws_cost(service):
    """Fake tool: returns AWS cost for a service"""
    costs = {
        "ec2": "$234.50",
        "rds": "$89.20",
        "s3": "$12.40"
    }
    return costs.get(service.lower(), "Service not found")

def send_slack_alert(message):
    """Fake tool: pretends to send a Slack message"""
    print(f"[SLACK] Alert sent: {message}")
    return "Alert delivered successfully"

def get_current_month():
    """Fake tool: returns current month"""
    import datetime
    return datetime.datetime.now().strftime("%B %Y")


# --- TOOL REGISTRY ---
# The agent picks tools from this dictionary by name
tools = {
    "get_aws_cost": get_aws_cost,
    "send_slack_alert": send_slack_alert,
    "get_current_month": get_current_month
}


# --- REACT LOOP ---
# Right now WE are deciding Thought/Action/Input manually.
# In Day 2, the LLM (Bedrock) will decide these automatically.

def run_react_loop(task):
    print(f"\n{'='*50}")
    print(f"TASK: {task}")
    print(f"{'='*50}\n")

    # Step 1
    thought = "I need to check the current month first"
    action = "get_current_month"
    action_input = None

    print(f"Thought: {thought}")
    print(f"Action: {action}")
    observation = tools[action]() if action_input is None else tools[action](action_input)
    print(f"Observation: {observation}\n")

    # Step 2
    thought = f"It's {observation}. Now I need to get EC2 cost."
    action = "get_aws_cost"
    action_input = "ec2"

    print(f"Thought: {thought}")
    print(f"Action: {action}({action_input})")
    observation = tools[action](action_input)
    print(f"Observation: {observation}\n")

    # Step 3
    thought = "EC2 cost is $234.50. That is over $200. I should send an alert."
    action = "send_slack_alert"
    action_input = f"EC2 cost this month is {observation} — exceeds $200 budget threshold"

    print(f"Thought: {thought}")
    print(f"Action: {action}")
    observation = tools[action](action_input)
    print(f"Observation: {observation}\n")

    # Final answer
    final = "EC2 cost is $234.50 which exceeds the $200 threshold. Slack alert has been sent."
    print(f"Final Answer: {final}")


# --- RUN IT ---
run_react_loop("Check EC2 cost this month and alert if it exceeds $200")