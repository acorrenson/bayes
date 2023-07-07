
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

        self._proba_x_A: Dict[str, int] = dict()
        self._proba_x_B: Dict[str, int] = dict()

        self._proba_A: float = 0
        self._proba_B: float = 0

        for point in self._points_A:
            self._nb_words_A += len(point)
        for point in self._points_B:
            self._nb_words_B += len(point)
        self._nb_words_total = self._nb_words_A + self._nb_words_B
        self._proba_A = self._nb_words_A / self._nb_words_total
        self._proba_B = self._nb_words_B / self._nb_words_total

    def count(self, x, label: bool):
        store = self._points_A if label else self._points_B
        c = 0
        for point in store:
            for w in point:
                if w == x:
                    c += 1
        return c

    def size(self, label: bool):
        if label:
            return self._nb_words_A
        else:
            return self._nb_words_B

    def proba(self, x: str, label: bool):
        store = self._proba_x_A if label else self._proba_x_B
        px = store.get(x)
        if px is None:
            px = self.count(x, label) / self.size(label)
            store[x] = px
        return px

    def proba_smooth(self, x, label: bool):
        store = self._proba_x_A if label else self._proba_x_B
        px = store.get(x)
        if px is None:
            px = (self.count(x, label) + 1) / (self.size(label) + 2)
            store[x] = px
        return px

    def predict(self, w: List[str]):
        pw_A = self._proba_A * \
            reduce(operator.mul, map(lambda x: self.proba(x, True), w))
        pw_B = self._proba_B * \
            reduce(operator.mul, map(lambda x: self.proba(x, False), w))
        return {'A': pw_A, 'B': pw_B}
