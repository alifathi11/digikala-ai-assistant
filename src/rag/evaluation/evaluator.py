import pandas as pd

from .metrics import (
    precision_at_k,
    recall_at_k,
    mrr
)


class RetrievalEvaluator:


    def __init__(
        self,
        retriever,
        dataset
    ):

        self.retriever = retriever
        self.dataset = dataset



    def evaluate(
        self,
        top_k=5
    ):

        precision_scores = []
        recall_scores = []
        mrr_scores = []


        for sample in self.dataset:


            results = self.retriever.retrieve(
                sample["query"],
                top_k=top_k
            )


            retrieved_ids = (
                results["id"]
                .tolist()
            )


            relevant_ids = (
                sample["relevant_ids"]
            )


            precision_scores.append(
                precision_at_k(
                    retrieved_ids,
                    relevant_ids,
                    top_k
                )
            )


            recall_scores.append(
                recall_at_k(
                    retrieved_ids,
                    relevant_ids,
                    top_k
                )
            )


            mrr_scores.append(
                mrr(
                    retrieved_ids,
                    relevant_ids
                )
            )


        return {
            "precision@k":
                sum(precision_scores)
                /
                len(precision_scores),

            "recall@k":
                sum(recall_scores)
                /
                len(recall_scores),

            "mrr":
                sum(mrr_scores)
                /
                len(mrr_scores)
        }