from pathlib import Path
import time

import pandas as pd


class ProductSearchEvaluationRunner:
    def __init__(self, metadata_retriever, review_evidence, reranker, metadata_k=50, reranker_k=12):
        self.metadata = metadata_retriever
        self.review_evidence = review_evidence
        self.reranker = reranker
        self.metadata_k = int(metadata_k)
        self.reranker_k = int(reranker_k)

    def _run_one(self, row):
        start = time.perf_counter()
        metadata_start = time.perf_counter()
        candidates = self.metadata.retrieve(row.query, top_k=self.metadata_k).reset_index(drop=True)
        metadata_latency = (time.perf_counter() - metadata_start) * 1000
        candidates["metadata_rank"] = range(1, len(candidates) + 1)
        shortlist = candidates.head(min(self.reranker_k, len(candidates))).copy().reset_index(drop=True)

        reviews, review_tel = self.review_evidence.retrieve(
            query=row.query,
            product_ids=shortlist["id"].astype(int).tolist(),
        )
        rankings, reranker_tel = self.reranker.rerank(row.query, shortlist, reviews)
        candidates = candidates.merge(rankings, on="id", how="left")
        candidates["query_id"] = str(row.query_id)
        candidates["query_type"] = str(row.query_type)
        candidates["query"] = str(row.query)
        candidates["split"] = str(row.split)

        telemetry = {
            "query_id": str(row.query_id),
            "query_type": str(row.query_type),
            "split": str(row.split),
            "metadata_latency_ms": float(metadata_latency),
            "total_latency_ms": float((time.perf_counter() - start) * 1000),
            **{f"review_{k}": v for k, v in review_tel.items()},
            **{f"reranker_{k}": v for k, v in reranker_tel.items()},
        }
        return candidates, telemetry

    def run(self, queries, candidates_path, telemetry_path, resume=True):
        candidates_path = Path(candidates_path)
        telemetry_path = Path(telemetry_path)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_parquet(candidates_path) if resume and candidates_path.exists() else pd.DataFrame()
        telemetry = pd.read_csv(telemetry_path) if resume and telemetry_path.exists() else pd.DataFrame()
        completed = set(existing["query_id"].astype(str).unique()) if len(existing) else set()
        candidate_parts = [existing] if len(existing) else []
        telemetry_parts = [telemetry] if len(telemetry) else []

        for row in queries.itertuples(index=False):
            if resume and str(row.query_id) in completed:
                continue
            frame, tel = self._run_one(row)
            candidate_parts.append(frame)
            telemetry_parts.append(pd.DataFrame([tel]))
            pd.concat(candidate_parts, ignore_index=True).to_parquet(candidates_path, index=False)
            pd.concat(telemetry_parts, ignore_index=True).to_csv(telemetry_path, index=False)

        return {
            "candidates": pd.concat(candidate_parts, ignore_index=True),
            "telemetry": pd.concat(telemetry_parts, ignore_index=True),
        }
