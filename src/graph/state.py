from operator import add
from typing import Annotated, List, Optional, TypedDict


class AgentState(TypedDict):
    """Graph state. Status lists use an `add` reducer so parallel scanners merge."""

    question: str
    question_status: Annotated[list, add]
    question_valid: bool
    intent: str  # FAQ | ORDER | OFF_TOPIC
    order_id: Optional[int]
    order: dict
    llm_output: str
    documents: List[dict]
    answer_status: Annotated[list, add]
    answer_valid: bool
