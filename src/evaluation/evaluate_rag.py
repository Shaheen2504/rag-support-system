"""
RAG System Evaluator

This module evaluates a RAG system using RAGAS metrics by sampling held-out questions,
generating answers, and measuring performance.
"""

import os

import polars as pl
from langchain.chat_models import init_chat_model
from loguru import logger
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

from src.config import settings
from src.graph.graph import create_workflow
from src.graph.utils import load_faiss_index


def setup_components():
    """Initialize all required components for RAG evaluation."""
    rag_app = create_workflow(load_faiss_index())
    llm = init_chat_model(
        settings.EVALUATION_LLM_MODEL, temperature=0.0, max_tokens=1000
    )
    return rag_app, LangchainLLMWrapper(llm)


def prepare_evaluation_data(rag_app):
    """
    Run the graph on a sample of the held-out test split (never indexed).
    Contexts are the documents the graph actually answered from.
    """
    test_df = pl.read_csv(settings.TEST_DATA_PATH)
    sample = test_df.sample(
        n=min(settings.EVALUATION_SAMPLE_SIZE, test_df.height),
        seed=settings.EVALUATION_RANDOM_SEED,
    )
    logger.info(f"Processing {sample.height} held-out questions...")

    dataset = []
    for i, row in enumerate(sample.iter_rows(named=True), 1):
        try:
            state = rag_app.invoke({"question": row["question"]})
            dataset.append(
                {
                    "user_input": row["question"],
                    "retrieved_contexts": [
                        f"question: {d['question']}\nanswer: {d['answer']}"
                        for d in state.get("documents", [])
                    ],
                    "response": state.get("llm_output", ""),
                    "reference": row["answer"],
                }
            )
            logger.info(f"Processed {i}/{sample.height}")
        except Exception as e:
            logger.info(f"Error processing question {i}: {e}")

    return dataset


def run_evaluation(dataset, evaluator_llm):
    """Run RAGAS evaluation and display results."""
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]

    logger.info("Running RAGAS evaluation...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    # Convert results to DataFrame for better display
    output_dir = settings.EVALUATION_OUTPUT_DIR
    results_df = results.to_pandas()
    # Save evaluation results
    results_html_path = os.path.join(output_dir, "evaluation_results.html")
    results_df.to_html(results_html_path, index=False)
    mean_scores = results_df.mean(numeric_only=True).round(4).to_frame(name="score")
    mean_scores_path = os.path.join(output_dir, "mean_scores.html")
    mean_scores.to_html(mean_scores_path)
    logger.info(f"Evaluation results saved to {output_dir}")


def main():
    """Main evaluation pipeline."""
    logger.info("Starting RAG evaluation...")

    # Setup
    rag_app, evaluator_llm = setup_components()

    # Prepare data and evaluate
    dataset = prepare_evaluation_data(rag_app)
    run_evaluation(dataset, evaluator_llm)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
