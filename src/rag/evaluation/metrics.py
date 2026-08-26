def precision_at_k(
    retrieved_ids,
    relevant_ids,
    k
):

    retrieved_ids = retrieved_ids[:k]

    hits = len(
        set(retrieved_ids)
        &
        set(relevant_ids)
    )

    return hits / k



def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k
):

    retrieved_ids = retrieved_ids[:k]

    hits = len(
        set(retrieved_ids)
        &
        set(relevant_ids)
    )

    if len(relevant_ids) == 0:
        return 0

    return hits / len(relevant_ids)



def mrr(
    retrieved_ids,
    relevant_ids
):

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if doc_id in relevant_ids:
            return 1 / rank


    return 0