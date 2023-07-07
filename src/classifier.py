
from dataclasses import dataclass
from functools import reduce
import operator
from typing import Dict, List


class Dataset2:
    """
      Datasets for binary classification of text
    """

    def __init__(self, points_A: List[List[str]], points_B: List[List[str]]):
        self._points_A = points_A
        self._points_B = points_B

        self._nb_words_A: int = 0
        self._nb_words_B: int = 0
        self._nb_words_total: int = 0

        self._occ_x_A: Dict[str, int] = dict()
        self._occ_x_B: Dict[str, int] = dict()

        self._proba_A: float = 0
        self._proba_B: float = 0

        for point in self._points_A:
            self._nb_words_A += len(point)
        for point in self._points_B:
            self._nb_words_B += len(point)
        self._nb_words_total = self._nb_words_A + self._nb_words_B
        self._proba_A = self._nb_words_A / self._nb_words_total
        self._proba_B = self._nb_words_B / self._nb_words_total

    def occurrences(self, w: str, label: bool):
        points = self._points_A if label else self._points_B
        memo = self._occ_x_A if label else self._occ_x_B
        occ_w = memo.get(w)
        if occ_w == None:
            c = 0
            for point in points:
                for p in point:
                    if w == p:
                        c += 1
            memo[w] = c
            return c
        return occ_w

    def size(self, label: bool):
        if label:
            return self._nb_words_A
        else:
            return self._nb_words_B


@dataclass
class Prediction:
    score_A: float
    score_B: float

    def get_class(self):
        if self.score_A > self.score_B:
            return True
        else:
            return False


class BayesianModel:
    def __init__(self, data: Dataset2):
        self._data = data

    def size(self, label: bool):
        return self._data.size(label)

    def occurences(self, w: str, label: bool):
        return self._data.occurrences(w, label)

    def proba_sentence(self, s: List[str], label: bool):
        assert False, "not implemented"

    def predict(self, s: List[str]) -> Prediction:
        return Prediction(self.proba_sentence(s, True), self.proba_sentence(s, False))


class NaiveBayesian(BayesianModel):
    def proba_sentence(self, s: List[str], label: bool):
        num = reduce(operator.mul, map(lambda w: self.occurences(w, label), s))
        den = self.size(label) ** len(s)
        # print(f'P({" ".join(s)} | {label}) = {num}/{den}')
        return num/den


class NaiveBayesianSmooth(BayesianModel):
    def proba_sentence(self, s: List[str], label: bool):
        num = reduce(operator.mul, map(
            lambda w: self.occurences(w, label) + 1, s))
        den = (self.size(label) + 2) ** len(s)
        # print(f'P({" ".join(s)} | {label}) = {num}/{den}')
        return num/den
