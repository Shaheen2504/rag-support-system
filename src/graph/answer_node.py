from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState
from src.graph.utils import get_llm

template = """You are a customer support assistant. Answer the question using only
the example support conversations below. Reply directly to the customer.
The examples contain placeholders like {{{{Order Number}}}} or {{{{Website URL}}}};
never copy a placeholder into your answer, describe the step in general terms instead.
If the examples do not cover the question, say you will connect them with a human agent.

Examples:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template=template)


def format_context(documents: list) -> str:
    return "\n\n".join(
        f"Q: {d['question']}\nA: {d['answer']}" for d in documents
    )


def answer_node(state: AgentState):
    """Generate answer node"""
    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {"question": state["question"], "context": format_context(state["documents"])}
    )
    return {"llm_output": answer}
