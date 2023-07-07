import re
from tokenizer import tokenize
from classifier import Dataset2
from generator import generate_data


def load(f):
    lines = None
    with open(f, "r") as f:
        lines = [line.rstrip() for line in f]
    return filter(lambda x: x != [], (list(tokenize(l)) for l in lines))


def print_prediction(w, p):
    pA = p['A']
    pB = p['B']
    print('scores:')
    print(f'A: {pA}')
    print(f'B: {pB}')
    if pA > pB:
        print(f'{w} is predicted to be of class A')
    else:
        print(f'{w} is predicted to be of class B')


def evaluate(Ntrain: int, Ntest: int):
    train_data_A = list(generate_data(Ntrain, True))
    train_data_B = list(generate_data(Ntrain, False))
    model = Dataset2(train_data_A, train_data_B)

    test_data_A = list(generate_data(Ntest, True))
    test_data_B = list(generate_data(Ntest, False))

    def check(w, label):
        pred = model.predict(w)
        plabel = 'A' if pred['A'] > pred['B'] else 'B'
        return label == plabel

    score_A = 0
    for point in test_data_A:
        score_A += int(check(point, 'A'))

    score_B = 0
    for point in test_data_B:
        score_B += int(check(point, 'B'))

    print("results:")
    print(f"A: {score_A}/{Ntest}")
    print(f"B: {score_B}/{Ntest}")
    print(f"overall: {100 * (score_A + score_B)/(2 * Ntest)}%")


if __name__ == "__main__":
    for _ in range(10):
        evaluate(1000, 10)
    # dataA = list(generate_data(10, True))
    # dataB = list(generate_data(10, False))
    # dataset = Dataset2(dataA, dataB)
    # for data in dataA:
    #     print(" ".join(data))
    # for data in dataB:
    #     print(" ".join(data))
    # w = input("sentence> ").strip().split(' ')
    # print_prediction(w, dataset.predict(w))
