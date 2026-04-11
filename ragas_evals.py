from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import pandas as pd
import json
import os

load_dotenv()


class RAGASEvaluator:
    """
    Evaluates the RAG pipeline using RAGAS metrics:
      - Faithfulness:       Is the answer grounded in the retrieved context?
      - Answer Relevancy:   Is the answer relevant to the question?
      - Context Precision:  Are the retrieved chunks actually useful?
      - Context Recall:     Did retrieval capture all necessary information?

    Context Recall requires a ground_truth answer for each question.
    The other three metrics are reference-free.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        llm = ChatOpenAI(model=model_name, temperature=0)
        self.ragas_llm = LangchainLLMWrapper(llm)
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        # Pass the same LLM to every metric
        for metric in self.metrics:
            metric.llm = self.ragas_llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_from_pipeline(
        self,
        questions: list[str],
        ground_truths: list[str],
        vector_store,
    ) -> dict:
        """
        Run the full RAG pipeline for each question, then evaluate with RAGAS.

        Args:
            questions:     List of test questions.
            ground_truths: Reference answers (one per question).
            vector_store:  A populated FAISS vector store from Embedding.create_vector_store().

        Returns:
            Dict with metric scores and a Pandas DataFrame of per-question results.
        """
        from retrieval import Retrieval
        from llm import LLM

        retrieval = Retrieval()
        llm = LLM()

        answers, contexts = [], []
        for question in questions:
            chunks = retrieval.retrieve(question, vector_store)
            context_texts = [chunk.page_content for chunk in chunks]
            answer = llm.generate_response(question, vector_store)
            answers.append(answer)
            contexts.append(context_texts)

        return self._run_ragas(questions, answers, contexts, ground_truths)

    def evaluate_from_data(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict:
        """
        Evaluate pre-collected QA data without re-running the pipeline.

        Args:
            questions:     Test questions.
            answers:       Model-generated answers.
            contexts:      Retrieved context chunks per question (list of lists).
            ground_truths: Reference answers.

        Returns:
            Dict with metric scores and a Pandas DataFrame of per-question results.
        """
        return self._run_ragas(questions, answers, contexts, ground_truths)

    def load_test_set(self, path: str) -> tuple[list, list, list, list]:
        """
        Load a JSON test set.  Expected format:

            [
              {
                "question":     "What is ...?",
                "ground_truth": "It is ...",
                "answer":       "Optional – omit to run the pipeline live",
                "contexts":     ["Optional", "pre-fetched", "chunks"]
              },
              ...
            ]

        Returns (questions, answers, contexts, ground_truths).
        answers / contexts are empty lists when not present in the file.
        """
        with open(path, "r") as f:
            data = json.load(f)

        questions     = [d["question"]     for d in data]
        ground_truths = [d["ground_truth"] for d in data]
        answers       = [d.get("answer",   "") for d in data]
        contexts      = [d.get("contexts", []) for d in data]
        return questions, answers, contexts, ground_truths

    def save_results(self, results: dict, output_path: str = "evaluation_results.csv"):
        """Persist per-question scores to a CSV file."""
        df: pd.DataFrame = results["dataframe"]
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_ragas(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> dict:
        dataset = Dataset.from_dict(
            {
                "question":     questions,
                "answer":       answers,
                "contexts":     contexts,
                "ground_truth": ground_truths,
            }
        )

        print("Running RAGAS evaluation…")
        result = evaluate(dataset=dataset, metrics=self.metrics)

        def _avg(val):
            """Normalize a score to a float (RAGAS may return a list or float)."""
            if isinstance(val, list):
                return sum(v for v in val if v is not None) / max(len(val), 1)
            return float(val)

        scores = {
            "faithfulness":      _avg(result["faithfulness"]),
            "answer_relevancy":  _avg(result["answer_relevancy"]),
            "context_precision": _avg(result["context_precision"]),
            "context_recall":    _avg(result["context_recall"]),
        }

        print("\n=== RAGAS Evaluation Results ===")
        for metric, score in scores.items():
            print(f"  {metric:<22}: {score:.4f}")

        return {"scores": scores, "dataframe": result.to_pandas()}


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Option A: evaluate from a pre-built JSON test set ------------------
    # (no live PDF / vector store needed)
    EXAMPLE_TEST_SET = [
        {
            "question": "What is the main topic of the document?",
            "ground_truth": "The document covers artificial intelligence fundamentals.",
            "answer": "The document is about artificial intelligence and its applications.",
            "contexts": [
                "Artificial intelligence (AI) is the simulation of human intelligence...",
                "Machine learning is a subset of AI that enables systems to learn from data...",
            ],
        },
        {
            "question": "What are the key benefits mentioned?",
            "ground_truth": "Increased efficiency and automation are the key benefits.",
            "answer": "The key benefits include increased efficiency, cost savings, and automation.",
            "contexts": [
                "One of the primary benefits of AI is increased operational efficiency...",
                "Automation powered by AI can significantly reduce costs...",
            ],
        },
    ]

    evaluator = RAGASEvaluator()
    results = evaluator.evaluate_from_data(
        questions=[d["question"]     for d in EXAMPLE_TEST_SET],
        answers  =[d["answer"]       for d in EXAMPLE_TEST_SET],
        contexts =[d["contexts"]     for d in EXAMPLE_TEST_SET],
        ground_truths=[d["ground_truth"] for d in EXAMPLE_TEST_SET],
    )
    evaluator.save_results(results, "evaluation_results.csv")

    # --- Option B: run the full pipeline live --------------------------------
    # from chunk import Chunk
    # from embedding import Embedding
    #
    # chunker = Chunk("your_document.pdf")
    # chunks  = chunker.create_chunks()
    # vector_store = Embedding().create_vector_store(chunks)
    #
    # questions, _, _, ground_truths = evaluator.load_test_set("test_set.json")
    # results = evaluator.evaluate_from_pipeline(questions, ground_truths, vector_store)
    # evaluator.save_results(results)