import re
from tokenizer import tokenize
from classifier import *
from generator import generate_data


def load(f):
    lines = None
    with open(f, "r") as f:
        lines = [line.rstrip() for line in f]
    return filter(lambda x: x != [], (list(tokenize(l)) for l in lines))


def evaluate_model(model: BayesianModel, Ntest: int, debug=False):
    test_data_A = list(generate_data(Ntest, True))
    test_data_B = list(generate_data(Ntest, False))

    def check(w, label):
        return label == model.predict(w).get_class()

    score_A = 0
    for point in test_data_A:
        score_A += int(check(point, True))

    score_B = 0
    for point in test_data_B:
        score_B += int(check(point, False))

    score = 100 * (score_A + score_B)/(2 * Ntest)
    if debug:
        print("results:")
        print(f"A: {score_A}/{Ntest}")
        print(f"B: {score_B}/{Ntest}")
        print(f"overall: {score}%")
    return score


if __name__ == "__main__":
    for i in range(10):
        print('-' * 100)
        print(f'Evaluation batch {i}')

        dataA = list(generate_data(500, True))
        dataB = list(generate_data(500, False))
        data = Dataset2(dataA, dataB)
        model_1 = NaiveBayesian(data)
        model_2 = NaiveBayesianSmooth(data)
        print(
            f'model 1 (Naive Bayesian)                     : {evaluate_model(model_1, 10)}')
        print(
            f'model 2 (Naive Bayesian + Laplace smoothing) : {evaluate_model(model_2, 10)}')
