from operator import add
from typing import Annotated, List, TypedDict


class AgentState(TypedDict):
    """Graph state. Status lists use an `add` reducer so parallel scanners merge."""

    question: str
    question_status: Annotated[list, add]
    question_valid: bool
    on_topic: str
    llm_output: str
    documents: List[dict]
    answer_status: Annotated[list, add]
    answer_valid: bool
