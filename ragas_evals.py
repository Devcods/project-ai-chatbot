"""
RAGAS EVALUATION SCRIPT
========================
This script scores how good your RAG app's answers are, using 4 metrics.
It does NOT run your app live — it reads pre-collected question/answer/context
data from test_set.json and asks an LLM "judge" to grade each one.

Think of it as 4 separate report cards, one per answer, then averaged.
"""

# 1. IMPORTS
# -----------------------------------------------------------------
# SingleTurnSample = one row of evaluation data (one question + its answer).
# EvaluationDataset = a collection of SingleTurnSample rows, the format
#                      RAGAS's evaluate() function expects.
from ragas import SingleTurnSample, EvaluationDataset
from ragas import evaluate                              # runs the metrics over the dataset

# The 4 metrics we're using to grade the RAG pipeline.
# Each one asks the judge LLM a different question about each answer.
from ragas.metrics import (
    faithfulness,       # "Is the answer backed up by the retrieved context, or did it make things up?"
    answer_relevancy,   # "Does the answer actually address the question that was asked?"
    context_precision,  # "Of the context that was retrieved, how much of it was actually useful?"
    context_recall,     # "Did the retrieval step pull in enough context to answer correctly?"
)

from ragas.llms import LangchainLLMWrapper   # lets RAGAS use a LangChain chat model as its judge
from langchain_openai import ChatOpenAI      # the actual OpenAI chat model class

from dotenv import load_dotenv               # reads OPENAI_API_KEY out of your .env file
import json                                  # to read test_set.json
import pandas as pd                          # to save the results as a CSV

load_dotenv()


# 2. THE JUDGE
# -----------------------------------------------------------------
# RAGAS metrics work by asking an LLM to grade each answer (e.g. "on a
# scale of 0-1, is this faithful to the context?"). That grading LLM is
# called the "judge". temperature=0 makes its grading as consistent and
# repeatable as possible (no creative randomness when scoring).
judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))


# 3. THE METRICS
# -----------------------------------------------------------------
# Every metric needs to know which LLM to use as its judge, so we assign
# it here before running anything.
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
for metric in metrics:
    metric.llm = judge


# 4. THE DATA
# -----------------------------------------------------------------
# test_set.json holds the questions, the answers your RAG app produced,
# the context chunks it retrieved, and the "correct" reference answer
# you wrote by hand. Each entry becomes one SingleTurnSample.
with open("test_set.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

samples = [
    SingleTurnSample(
        user_input=case["question"],           # the question that was asked
        retrieved_contexts=case["contexts"],    # the chunks your retriever pulled from the PDF
        response=case["answer"],                # what your RAG app answered
        reference=case["ground_truth"],         # the answer you'd consider "correct"
    )
    for case in test_cases
]

dataset = EvaluationDataset(samples=samples)


# 5. RUN + READ
# -----------------------------------------------------------------
# This sends every sample to the judge LLM once per metric (4 metrics x
# N questions = 4N judge calls), so it can take a little while and will
# use OpenAI credits.
result = evaluate(dataset=dataset, metrics=metrics)

print(result)

# Save a row-by-row breakdown (one row per question, one column per
# metric) so you can see which specific answers scored low, not just
# the overall average.
result.to_pandas().to_csv("evaluation_results.csv", index=False)
print("\nSaved detailed scores to evaluation_results.csv")
