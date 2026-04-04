from dataclasses import dataclass, field
from enum import StrEnum

from src.models.base import PromptResponse
 
 
class Agent(StrEnum):
    PLANNING = "planning"
    SIMULATION = "simulation"
    PLAN_REFINEMENT = "plan_refinement"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    MATH_GENERATION = "math_generation"
 
 
@dataclass
class AgentCall:
    agent: Agent
    task_id: str
    time_taken: float
    prompt_tokens: int
    completion_tokens: int
 
 
@dataclass
class MetricsState:
    calls: list[AgentCall] = field(default_factory=list)
 
    def record(self, agent: Agent, task_id: str, response: PromptResponse) -> None:
        self.calls.append(AgentCall(
            agent=agent,
            task_id=task_id,
            time_taken=response["time_taken"],
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
        ))
 
    # --- aggregates across ALL agents ---
 
    @property
    def total_time(self) -> float:
        return sum(c.time_taken for c in self.calls)
 
    @property
    def total_input_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)
 
    @property
    def total_output_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)
 
    @property
    def total_api_calls(self) -> int:
        return len(self.calls)
 
    # --- per-agent breakdown ---
 
    def _calls_for(self, agent: Agent) -> list[AgentCall]:
        return [c for c in self.calls if c.agent == agent]
 
    def agent_time(self, agent: Agent) -> float:
        return sum(c.time_taken for c in self._calls_for(agent))
 
    def agent_input_tokens(self, agent: Agent) -> int:
        return sum(c.prompt_tokens for c in self._calls_for(agent))
 
    def agent_output_tokens(self, agent: Agent) -> int:
        return sum(c.completion_tokens for c in self._calls_for(agent))
 
    def agent_api_calls(self, agent: Agent) -> int:
        return len(self._calls_for(agent))
 
    # --- per-task breakdown ---
 
    def _calls_for_task(self, task_id: str) -> list[AgentCall]:
        return [c for c in self.calls if c.task_id == task_id]
 
    def task_time(self, task_id: str) -> float:
        return sum(c.time_taken for c in self._calls_for_task(task_id))
 
    def task_api_calls(self, task_id: str) -> int:
        return len(self._calls_for_task(task_id))
 
    # --- reporting ---
 
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "METRICS SUMMARY",
            "=" * 50,
            f"{'Total API calls':<30} {self.total_api_calls}",
            f"{'Total time (s)':<30} {self.total_time:.2f}",
            f"{'Total input tokens':<30} {self.total_input_tokens}",
            f"{'Total output tokens':<30} {self.total_output_tokens}",
            "",
            f"{'Agent':<25} {'Calls':>6} {'Time (s)':>10} {'In tokens':>12} {'Out tokens':>12}",
            "-" * 70,
        ]
        for agent in Agent:
            lines.append(
                f"{agent:<25}"
                f" {self.agent_api_calls(agent):>6}"
                f" {self.agent_time(agent):>10.2f}"
                f" {self.agent_input_tokens(agent):>12}"
                f" {self.agent_output_tokens(agent):>12}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)