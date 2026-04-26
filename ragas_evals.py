# Import Dataset so we can create data in the format RAGAS needs
from datasets import Dataset

# Import evaluate function from RAGAS
from ragas import evaluate

# Import the RAGAS metrics we want to test
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# This wrapper lets RAGAS use a LangChain LLM
from ragas.llms import LangchainLLMWrapper

# ChatOpenAI is used to connect OpenAI model through LangChain
from langchain_openai import ChatOpenAI

# Loads the API key from .env file
from dotenv import load_dotenv

# Used to save results in CSV format
import pandas as pd

# Used to read JSON test files
import json


# Load environment variables from .env file
load_dotenv()


# This class handles the RAGAS evaluation
class RAGASEvaluator:

    # This runs when we create the evaluator object
    def __init__(self, model_name="gpt-3.5-turbo"):

        # Create the OpenAI model with temperature 0 for stable answers
        llm = ChatOpenAI(model=model_name, temperature=0)

        # Wrap the LLM so RAGAS can use it
        ragas_llm = LangchainLLMWrapper(llm)

        # Store all the metrics we want to calculate
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        # Give the same LLM to each metric
        for metric in self.metrics:
            metric.llm = ragas_llm

    # This function evaluates already collected data
    def evaluate_data(self, questions, answers, contexts, ground_truths):

        # Create a dataset in the format RAGAS expects
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Print message before evaluation starts
        print("Running evaluation...")

        # Run RAGAS evaluation using the dataset and metrics
        result = evaluate(dataset=dataset, metrics=self.metrics)

        # Store average scores for each metric
        scores = {
            "faithfulness": self.get_average(result["faithfulness"]),
            "answer_relevancy": self.get_average(result["answer_relevancy"]),
            "context_precision": self.get_average(result["context_precision"]),
            "context_recall": self.get_average(result["context_recall"]),
        }

        # Print the final scores
        print("\nRAGAS Scores:")

        # Print each metric score one by one
        for name, score in scores.items():
            print(f"{name}: {score:.4f}")

        # Return both scores and full dataframe
        return {
            "scores": scores,
            "dataframe": result.to_pandas()
        }

    # This function runs the full RAG pipeline first, then evaluates it
    def evaluate_pipeline(self, questions, ground_truths, vector_store):

        # Import retrieval class from our project
        from retrieval import Retrieval

        # Import LLM class from our project
        from llm import LLM

        # Create retrieval object
        retrieval = Retrieval()

        # Create LLM object
        llm = LLM()

        # Empty list to store generated answers
        answers = []

        # Empty list to store retrieved contexts
        contexts = []

        # Loop through every question
        for question in questions:

            # Retrieve relevant documents from vector store
            docs = retrieval.retrieve(question, vector_store)

            # Empty list for storing text from documents
            context_texts = []

            # Take text from every retrieved document
            for doc in docs:
                context_texts.append(doc.page_content)

            # Generate answer using our RAG pipeline
            answer = llm.generate_response(question, vector_store)

            # Save generated answer
            answers.append(answer)

            # Save retrieved context text
            contexts.append(context_texts)

        # Evaluate the generated answers and contexts
        return self.evaluate_data(
            questions,
            answers,
            contexts,
            ground_truths
        )

    # This function loads test data from a JSON file
    def load_test_set(self, path):

        # Open the JSON file
        with open(path, "r") as file:

            # Read JSON data
            data = json.load(file)

        # Empty list for questions
        questions = []

        # Empty list for generated answers
        answers = []

        # Empty list for retrieved contexts
        contexts = []

        # Empty list for correct reference answers
        ground_truths = []

        # Loop through each item in the JSON file
        for item in data:

            # Add question
            questions.append(item["question"])

            # Add correct answer
            ground_truths.append(item["ground_truth"])

            # Add generated answer if it exists, otherwise add empty string
            answers.append(item.get("answer", ""))

            # Add contexts if they exist, otherwise add empty list
            contexts.append(item.get("contexts", []))

        # Return all lists
        return questions, answers, contexts, ground_truths

    # This function saves final result dataframe into a CSV file
    def save_results(self, results, output_path="evaluation_results.csv"):

        # Get dataframe from results dictionary
        df = results["dataframe"]

        # Save dataframe as CSV
        df.to_csv(output_path, index=False)

        # Print where the file was saved
        print(f"Saved results to {output_path}")

    # This helper function calculates average score
    def get_average(self, values):

        # If values are in list form, calculate average manually
        if isinstance(values, list):

            # Remove None values
            clean_values = [v for v in values if v is not None]

            # If no valid values exist, return 0
            if len(clean_values) == 0:
                return 0

            # Return average of valid values
            return sum(clean_values) / len(clean_values)

        # If value is already one number, convert it to float
        return float(values)


# This part only runs when this file is executed directly
if __name__ == "__main__":

    # Create evaluator object
    evaluator = RAGASEvaluator()

    # Load questions, answers, contexts, and ground truths from JSON file
    questions, answers, contexts, ground_truths = evaluator.load_test_set(
        "test_set.json"
    )

    # Run evaluation on the loaded test data
    results = evaluator.evaluate_data(
        questions,
        answers,
        contexts,
        ground_truths
    )

    # Save results into CSV file
    evaluator.save_results(results)