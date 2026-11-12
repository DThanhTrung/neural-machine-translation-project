from pathlib import Path
from collections import Counter
from collections.abc import Iterable
from math import exp, log

class CorpusBLEU:
    def __init__(
        self, 
        max_order: int = 4,
        smoothing_eps: float = 1e-10
    ) -> None:
        if not isinstance(max_order, int) or max_order <= 0:
            raise ValueError("max_order phải là số nguyên dương!")

        self.max_order = max_order
        self.smoothing_eps = smoothing_eps


    @staticmethod
    def _get_ngrams(tokens: list[str], order: int) -> Counter:
        return Counter(
            tuple(tokens[index : (index + order)])
            for index in range(len(tokens) - order + 1)
        )


    @classmethod
    def _calculate_precision(
        cls,
        prediction: str,
        reference: str,
        order: int,
    ) -> tuple[int, int]:
        prediction_tokens = prediction.lower().split()
        reference_tokens = reference.lower().split()

        prediction_counts = cls._get_ngrams(prediction_tokens, order)
        reference_counts = cls._get_ngrams(reference_tokens, order)

        numerator = sum(
            min(count, reference_counts[ngram])
            for ngram, count in prediction_counts.items()
        )
        denominator = sum(prediction_counts.values())

        return numerator, denominator


    def calculate(
        self,
        predictions: Iterable[str],
        references: Iterable[str],
    ) -> float:
        predictions = list(predictions)
        references = list(references)

        if len(predictions) != len(references):
            raise ValueError("Số câu dự đoán phải khớp với số câu tham chiếu!")

        if not predictions:
            return 0.0


        log_precision_sum = 0.0
        for order in range(1, self.max_order + 1):
            numerator = denominator = 0

            for prediction, reference in zip(predictions, references):
                matched, total = self._calculate_precision(
                    prediction, reference, order
                )

                numerator += matched
                denominator += total

            log_precision_sum += log(
                numerator / denominator if min(numerator, denominator) > 0
                else self.smoothing_eps
            )

        prediction_length = sum(
            len(prediction.split()) for prediction in predictions
        )

        reference_length = sum(
            len(reference.split()) for reference in references
        )

        if prediction_length == 0:
            return 0.0

        brevity_penalty = (
            1.0
            if prediction_length > reference_length
            else exp(1 - reference_length / prediction_length)
        )

        return brevity_penalty * exp(log_precision_sum / self.max_order)