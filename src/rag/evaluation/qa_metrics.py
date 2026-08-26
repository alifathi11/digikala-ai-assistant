import numpy as np


def citation_validity(
    evidence_ids,
    retrieved_ids,
):
    evidence_ids = {
        int(x)
        for x in evidence_ids
    }

    retrieved_ids = {
        int(x)
        for x in retrieved_ids
    }

    # Citation validity asks whether every cited ID is valid.
    # An empty citation set is valid by definition: it contains no
    # fabricated/out-of-retrieval IDs. Whether an answer is allowed to
    # have no evidence is enforced separately by the QA response validator
    # (`insufficient_evidence` vs supported answer).
    if not evidence_ids:
        return 1.0

    return float(
        evidence_ids.issubset(
            retrieved_ids
        )
    )


def evidence_precision(
    evidence_ids,
    relevant_ids,
):
    predicted = {
        int(x)
        for x in evidence_ids
    }

    relevant = {
        int(x)
        for x in relevant_ids
    }

    if not predicted:
        return 0.0

    return (
        len(
            predicted
            & relevant
        )
        / len(predicted)
    )


def evidence_recall(
    evidence_ids,
    relevant_ids,
):
    predicted = {
        int(x)
        for x in evidence_ids
    }

    relevant = {
        int(x)
        for x in relevant_ids
    }

    if not relevant:
        return 0.0

    return (
        len(
            predicted
            & relevant
        )
        / len(relevant)
    )


def evidence_f1(
    evidence_ids,
    relevant_ids,
):
    precision = evidence_precision(
        evidence_ids,
        relevant_ids,
    )

    recall = evidence_recall(
        evidence_ids,
        relevant_ids,
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def retrieval_evidence_recall(
    retrieved_ids,
    relevant_ids,
):
    retrieved = {
        int(x)
        for x in retrieved_ids
    }

    relevant = {
        int(x)
        for x in relevant_ids
    }

    if not relevant:
        return 0.0

    return (
        len(
            retrieved
            & relevant
        )
        / len(relevant)
    )


def percentile(
    values,
    q,
):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if not values:
        return None

    return float(
        np.quantile(
            values,
            q,
        )
    )
