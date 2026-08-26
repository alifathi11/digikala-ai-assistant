import math


def precision_at_k(
    retrieved_ids,
    relevant_ids,
    k
):
    retrieved_ids = retrieved_ids[:k]

    hits = len(
        set(retrieved_ids)
        & set(relevant_ids)
    )

    return hits / k



def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k
):
    relevant_ids = set(relevant_ids)

    if len(relevant_ids) == 0:
        return 0.0

    hits = len(
        set(retrieved_ids[:k])
        & relevant_ids
    )

    return hits / len(relevant_ids)



def hit_rate_at_k(
    retrieved_ids,
    relevant_ids,
    k
):
    return float(
        bool(
            set(retrieved_ids[:k])
            & set(relevant_ids)
        )
    )



def reciprocal_rank_at_k(
    retrieved_ids,
    relevant_ids,
    k
):
    relevant_ids = set(relevant_ids)

    for rank, doc_id in enumerate(
        retrieved_ids[:k],
        start=1
    ):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0



def mrr(
    retrieved_ids,
    relevant_ids
):
    return reciprocal_rank_at_k(
        retrieved_ids,
        relevant_ids,
        len(retrieved_ids)
    )



def average_precision_at_k(
    retrieved_ids,
    relevant_ids,
    k
):
    relevant_ids = set(relevant_ids)

    if len(relevant_ids) == 0:
        return 0.0

    hits = 0
    score = 0.0

    for rank, doc_id in enumerate(
        retrieved_ids[:k],
        start=1
    ):
        if doc_id in relevant_ids:
            hits += 1
            score += hits / rank

    denominator = min(
        len(relevant_ids),
        k
    )

    return (
        score / denominator
        if denominator
        else 0.0
    )



def ndcg_at_k(
    retrieved_ids,
    relevant_ids,
    k
):
    relevant_ids = set(relevant_ids)

    dcg = 0.0

    for rank, doc_id in enumerate(
        retrieved_ids[:k],
        start=1
    ):
        relevance = float(
            doc_id in relevant_ids
        )

        dcg += (
            relevance
            / math.log2(rank + 1)
        )

    ideal_hits = min(
        len(relevant_ids),
        k
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return (
        dcg / idcg
        if idcg > 0
        else 0.0
    )
