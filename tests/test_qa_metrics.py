from src.rag.evaluation.qa_metrics import (
    citation_validity,
    evidence_precision,
    evidence_recall,
    evidence_f1,
)


def test_empty_citations_are_valid():
    assert citation_validity([], [1, 2]) == 1.0


def test_invalid_citation_is_rejected():
    assert citation_validity([3], [1, 2]) == 0.0


def test_evidence_metrics():
    predicted = [1, 2]
    relevant = [2, 3]

    assert evidence_precision(predicted, relevant) == 0.5
    assert evidence_recall(predicted, relevant) == 0.5
    assert evidence_f1(predicted, relevant) == 0.5
