import math

import pandas as pd


def rank_policy(candidates, policy):
    frame = candidates.copy()
    kind = str(policy["kind"])

    if kind == "metadata":
        ranked = frame.sort_values("metadata_rank")
    else:
        reranker_k = int(policy.get("reranker_k", 12))
        ranked = frame[frame["metadata_rank"] <= reranker_k].copy()
        ranked["llm_match_score"] = pd.to_numeric(ranked["llm_match_score"], errors="coerce").fillna(0.0)
        ranked["llm_norm"] = ranked["llm_match_score"] / 5.0
        ranked["metadata_score"] = pd.to_numeric(ranked["metadata_score"], errors="coerce").fillna(0.0)

        if kind == "llm_only":
            ranked["_score"] = ranked["llm_norm"]
            ranked = ranked.sort_values(["_score", "metadata_score"], ascending=[False, False])
        elif kind in {"weighted", "llm_tiered"}:
            mw = float(policy["metadata_weight"])
            lw = float(policy["llm_weight"])
            ranked["_score"] = (mw * ranked["metadata_score"] + lw * ranked["llm_norm"]) / (mw + lw)
            if kind == "llm_tiered":
                ranked = ranked.sort_values(["llm_match_score", "_score", "metadata_score"], ascending=[False, False, False])
            else:
                ranked = ranked.sort_values(["_score", "llm_match_score", "metadata_score"], ascending=[False, False, False])
        else:
            raise ValueError(f"Unknown policy kind: {kind}")

    ranked = ranked.reset_index(drop=True)
    ranked["policy_rank"] = range(1, len(ranked) + 1)
    return ranked


def _dcg(grades):
    return float(sum((2 ** int(grade) - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1)))


def _query_metrics(ranked, qrels, ks, relevant_threshold):
    grade_map = qrels.set_index("id")["relevance_grade"].to_dict()
    relevant = {int(pid) for pid, grade in grade_map.items() if int(grade) >= int(relevant_threshold)}
    ranked_ids = ranked["id"].astype(int).tolist()
    max_k = max(ks)
    first = next((rank for rank, pid in enumerate(ranked_ids[:max_k], start=1) if pid in relevant), None)
    output = {f"mrr@{max_k}": 1.0 / first if first else 0.0}

    for k in ks:
        ids = ranked_ids[:k]
        hits = [int(pid in relevant) for pid in ids]
        output[f"precision@{k}"] = sum(hits) / float(k)
        output[f"hit_rate@{k}"] = float(any(hits))
        output[f"pool_recall@{k}"] = sum(hits) / len(relevant) if relevant else 0.0
        actual = [int(grade_map.get(pid, 0)) for pid in ids] + [0] * (k - len(ids))
        ideal = sorted([int(value) for value in grade_map.values()], reverse=True)[:k]
        ideal += [0] * (k - len(ideal))
        ideal_dcg = _dcg(ideal)
        output[f"ndcg@{k}"] = _dcg(actual) / ideal_dcg if ideal_dcg else 0.0
    return output


def evaluate_policies(candidates, qrels, policies, ks=(1,3,5,10), relevant_threshold=2):
    rows = []
    qrel_groups = {str(qid): group for qid, group in qrels.groupby("query_id", sort=False)}
    for query_id, group in candidates.groupby("query_id", sort=False):
        query_id = str(query_id)
        if query_id not in qrel_groups:
            continue
        for policy in policies:
            ranked = rank_policy(group, policy)
            rows.append({
                "query_id": query_id,
                "query_type": str(group["query_type"].iloc[0]),
                "split": str(group["split"].iloc[0]),
                "policy": policy["name"],
                **_query_metrics(ranked, qrel_groups[query_id], ks, relevant_threshold),
            })
    return pd.DataFrame(rows)


def aggregate_metrics(per_query):
    metric_columns = [column for column in per_query.columns if "@" in column]
    return per_query.groupby(["split", "policy"], as_index=False)[metric_columns].mean()


def select_best_policy(aggregate, metric="ndcg@10"):
    dev = aggregate[aggregate["split"] == "dev"].copy()
    if len(dev) == 0:
        raise ValueError("No dev rows.")
    return str(dev.sort_values([metric, "mrr@10", "hit_rate@5"], ascending=[False, False, False]).iloc[0]["policy"])


def candidate_recall(candidates, qrels, relevant_threshold=2, shortlist_k=12):
    rows = []
    for query_id, qrel_group in qrels.groupby("query_id", sort=False):
        relevant = set(qrel_group[qrel_group["relevance_grade"] >= relevant_threshold]["id"].astype(int))
        group = candidates[candidates["query_id"].astype(str) == str(query_id)]
        shortlist = set(group[group["metadata_rank"] <= shortlist_k]["id"].astype(int))
        rows.append({
            "query_id": str(query_id),
            "query_type": str(qrel_group["query_type"].iloc[0]),
            "split": str(qrel_group["split"].iloc[0]),
            "relevant_in_judged_pool": len(relevant),
            "candidate_recall@12": len(relevant & shortlist) / len(relevant) if relevant else 0.0,
            "candidate_hit@12": float(bool(relevant & shortlist)),
        })
    return pd.DataFrame(rows)


def failure_analysis(candidates, qrels, selected_policy, relevant_threshold=2, shortlist_k=12):
    rows = []
    for query_id, qrel_group in qrels.groupby("query_id", sort=False):
        query_id = str(query_id)
        group = candidates[candidates["query_id"].astype(str) == query_id]
        if len(group) == 0:
            continue
        relevant = set(qrel_group[qrel_group["relevance_grade"] >= relevant_threshold]["id"].astype(int))
        shortlist = set(group[group["metadata_rank"] <= shortlist_k]["id"].astype(int))
        ranked = rank_policy(group, selected_policy)
        top5 = set(ranked.head(5)["id"].astype(int))
        grade_map = qrel_group.set_index("id")["relevance_grade"].to_dict()
        top1_id = int(ranked.iloc[0]["id"]) if len(ranked) else None
        top1_grade = int(grade_map.get(top1_id, 0))

        if not relevant:
            failure = "no_relevant_in_judged_pool"
        elif not (relevant & shortlist):
            failure = "candidate_retrieval_miss"
        elif not (relevant & top5):
            failure = "reranker_miss"
        elif top1_grade < relevant_threshold:
            failure = "top_rank_error"
        else:
            failure = "ok"

        rows.append({
            "query_id": query_id,
            "query_type": str(group["query_type"].iloc[0]),
            "split": str(group["split"].iloc[0]),
            "query": str(group["query"].iloc[0]),
            "failure_type": failure,
            "top1_id": top1_id,
            "top1_grade": top1_grade,
            "top1_title": str(ranked.iloc[0]["title_fa"]) if len(ranked) else "",
        })
    return pd.DataFrame(rows)
