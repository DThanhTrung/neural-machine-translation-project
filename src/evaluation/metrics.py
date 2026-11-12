from pathlib import Path
from collections import Counter
from math import exp, log


def _get_ngrams(
    tokens: list[str], 
    order: int
) -> Counter:
    return Counter(
        tuple(tokens[idx : (idx + order)])
        for idx in range(len(tokens) - order + 1)
    )


def _calculate_precision(
    prediction: list[str],
    reference: list[str],
    order: int,
) -> tuple[int, int] :
    prediction_counts = _get_ngrams(prediction, order)
    reference_counts = _get_ngrams(reference, order)

    numerator = sum(
        min(count, reference_counts[ngram])
        for ngram, count in prediction_counts.items()
    )
    denominator = sum(prediction_counts.values())

    return numerator, denominator


def _calculate_corpus_level_bleu(
    predictions: list[list[str]],
    references: list[list[str]],
    max_order: int = 4,
    smoothing_eps: float = 1e-10,
) -> float :
    predictions = list(predictions)
    references = list(references)

    if len(predictions) != len(references):
        raise ValueError("Số câu dự đoán phải khớp với số câu tham chiếu!")

    if not predictions:
        return 0.0

    log_precision_sum = 0.0
    for order in range(1, max_order + 1):
        numerator = denominator = 0

        for prediction, reference in zip(predictions, references):
            matched, total = _calculate_precision(
                prediction, reference, order
            )

            numerator += matched
            denominator += total

        log_precision_sum += log(
            numerator / denominator if min(numerator, denominator) > 0
            else smoothing_eps
        )

    prediction_length = sum(
        len(prediction) for prediction in predictions
    )

    reference_length = sum(
        len(reference) for reference in references
    )

    if prediction_length == 0:
        return 0.0

    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else exp(1 - reference_length / prediction_length)
    )

    return brevity_penalty * exp(log_precision_sum / max_order)


def calculate_bleu(
    predictions: list[list[str]],
    references: list[list[str]],
    max_order: int = 4,
    smoothing_eps: float = 1e-10,
) -> float :
    if not isinstance(max_order, int) or max_order <= 0:
        raise ValueError("max_order phải là số nguyên dương!")
    if smoothing_eps <= 0:
        raise ValueError("smoothing_eps phải lớn hơn 0!")

    return _calculate_corpus_level_bleu(
        predictions=predictions,
        references=references,
        max_order=max_order,
        smoothing_eps=smoothing_eps
    )

