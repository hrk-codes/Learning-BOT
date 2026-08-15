from dataclasses import dataclass, field
from typing import Any
from typing import Literal


AgentStatus = Literal["running", "completed", "max_iterations", "error"]


@dataclass
class AgentTraceStep:
    iteration: int
    observation: str
    action: str
    status: str
    content: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    retrieval_query: str | None = None
    retrieved_chunks: list[dict[str, Any]] | None = None
    retrieval_latency_seconds: float = 0.0
    elapsed_seconds: float = 0.0


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
    llm_call_count: int = 0
    tool_call_count: int = 0
    total_tool_latency_seconds: float = 0.0
    rag_retrieval_count: int = 0
    total_rag_latency_seconds: float = 0.0
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)

    def record_observation(self, observation: str) -> None:
        self.observations.append(observation)

    def record_action_result(self, result: str) -> None:
        self.action_results.append(result)

    def record_trace(self, step: AgentTraceStep) -> None:
        self.trace.append(step)
