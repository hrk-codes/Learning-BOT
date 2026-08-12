from dataclasses import dataclass, field
from typing import Literal


AgentStatus = Literal["running", "completed", "max_iterations", "error"]


@dataclass
class AgentTraceStep:
    iteration: int
    observation: str
    action: str
    status: str
    content: str


@dataclass
class AgentState:
    goal: str
    max_iterations: int
    iteration_count: int = 0
    status: AgentStatus = "running"
    final_answer: str = ""
    observations: list[str] = field(default_factory=list)
    action_results: list[str] = field(default_factory=list)
    trace: list[AgentTraceStep] = field(default_factory=list)

    def record_observation(self, observation: str) -> None:
        self.observations.append(observation)

    def record_action_result(self, result: str) -> None:
        self.action_results.append(result)

    def record_trace(self, step: AgentTraceStep) -> None:
        self.trace.append(step)
