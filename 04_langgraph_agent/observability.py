# MLflow observability for LangGraph agents
# Import this in any agent to get full tracking

import mlflow
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "multi-agent-oncall")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


class NodeTracker:
    """
    Tracks a single LangGraph node execution in MLflow.
    Handles multiple calls to same node (e.g. call_llm loops).

    Usage:
        with NodeTracker("cost_agent", "analyze_costs") as t:
            result = llm.invoke(...)
            t.log_tokens(input_tokens, output_tokens)
    """

    # Class-level call counter — tracks how many times each node ran
    _call_counts = {}

    def __init__(self, agent_name: str, node_name: str):
        self.agent_name = agent_name
        self.node_name  = node_name
        self.start_time = None

        # Increment call count for this node
        key = f"{agent_name}/{node_name}"
        NodeTracker._call_counts[key] = NodeTracker._call_counts.get(key, 0) + 1
        self.call_count = NodeTracker._call_counts[key]

    def __enter__(self):
        self.start_time = time.time()
        print(f"   [MLflow] Tracking: {self.agent_name}/{self.node_name} (call #{self.call_count})")
        return self

    def log_tokens(self, input_tokens: int, output_tokens: int, model: str = "claude-sonnet-4"):
        """Track LLM token usage and cost."""
        total = input_tokens + output_tokens
        cost  = (input_tokens * 0.003 + output_tokens * 0.015) / 1000

        # Use step=call_count so multiple calls don't clash
        mlflow.log_metrics({
            f"{self.node_name}_input_tokens":  input_tokens,
            f"{self.node_name}_output_tokens": output_tokens,
            f"{self.node_name}_total_tokens":  total,
            f"{self.node_name}_cost_usd":      round(cost, 6),
        }, step=self.call_count)

    def log_routing(self, decision: str):
        """Track routing decision — uses metric not param to allow multiple calls."""
        mlflow.log_metric(
            f"{self.node_name}_is_tool_call",
            1 if decision == "tool_call" else 0,
            step=self.call_count
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        success    = exc_type is None

        mlflow.log_metrics({
            f"{self.node_name}_latency_ms": round(latency_ms, 2),
            f"{self.node_name}_success":    1 if success else 0,
        }, step=self.call_count)

        if exc_val:
            # Use unique key per call to avoid param clash
            mlflow.log_param(
                f"{self.node_name}_error_call{self.call_count}",
                str(exc_val)[:300]
            )

        return False  # don't suppress exceptions


class AgentRunTracker:
    """
    Tracks a complete agent run — wraps all nodes.
    Resets NodeTracker call counts at start of each run.

    Usage:
        with AgentRunTracker("oncall_agent", channel="#git-script-repo") as run:
            result = graph.invoke(initial_state)
            run.log_outcome("success", alert_sent=1)
    """

    def __init__(self, run_name: str, channel: str = "unknown", intent: str = "unknown"):
        self.run_name   = run_name
        self.channel    = channel
        self.intent     = intent
        self.start_time = None

    def __enter__(self):
        # Reset node call counters for fresh run
        NodeTracker._call_counts = {}

        self.start_time = time.time()
        self._run = mlflow.start_run(
            run_name=f"{self.run_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow.log_params({
            "agent_name": self.run_name,
            "channel":    self.channel,
            "intent":     self.intent,
            "started_at": datetime.utcnow().isoformat(),
        })
        return self

    def log_intent(self, intent: str, confidence: float):
        """Call after intent detection."""
        mlflow.log_params({
            "detected_intent":    intent,
            "intent_confidence":  str(round(confidence, 4))
        })

    def log_outcome(self, outcome: str, **kwargs):
        """
        Call after agent completes.
        outcome: success | failed | hitl_approved | hitl_rejected
        kwargs: any extra metrics (alert_sent, pr_merged, etc.)
        """
        total_ms = (time.time() - self.start_time) * 1000

        mlflow.log_params({"outcome": outcome})
        mlflow.log_metrics({
            "total_latency_ms": round(total_ms, 2),
            "success": 1 if outcome in ("success", "hitl_approved") else 0,
        })

        # Log extra kwargs
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, str(v)[:250])

    def log_state(self, state: dict):
        """Log final graph state — skips messages list."""
        loggable = {
            k: str(v)[:250]
            for k, v in state.items()
            if k not in ("messages",) and v is not None and v != ""
        }
        if loggable:
            mlflow.log_params(loggable)

    def log_error(self, error: str):
        """Call if agent fails."""
        mlflow.log_param("error", str(error)[:300])
        mlflow.log_metric("success", 0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.log_error(str(exc_val))

        mlflow.set_tags({
            "agent":   self.run_name,
            "channel": self.channel,
            "intent":  self.intent,
        })
        mlflow.end_run()
        return False  # don't suppress exceptions