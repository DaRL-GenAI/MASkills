"""Token usage tracking and cost estimation."""

from typing import Dict


class TokenTracker:
    """Track token usage and estimate costs for LLM API calls."""

    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        input_price: float = None,
        output_price: float = None,
    ):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

        if input_price is not None and output_price is not None:
            self.pricing = {"input": input_price, "output": output_price}
        elif model in self.PRICING:
            self.pricing = self.PRICING[model]
        else:
            self.pricing = self.PRICING["gpt-4o-mini"]

    def add_usage(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_stats(self) -> Dict:
        total_tokens = self.input_tokens + self.output_tokens
        input_cost = (self.input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * self.pricing["output"]
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "cost_usd": input_cost + output_cost,
        }

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        return input_cost + output_cost

    def get_summary_string(self) -> str:
        stats = self.get_stats()
        return (
            f"Token Usage Summary ({stats['model']}):\n"
            f"  Input tokens:  {stats['input_tokens']:,}\n"
            f"  Output tokens: {stats['output_tokens']:,}\n"
            f"  Total tokens:  {stats['total_tokens']:,}\n"
            f"  Estimated cost: ${stats['cost_usd']:.4f}"
        )
