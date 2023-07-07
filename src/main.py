import re
from tokenizer import tokenize


def load(f):
    lines = None
    with open(f, "r") as f:
        lines = [line.rstrip() for line in f]
    return filter(lambda x: x != [], (list(tokenize(l)) for l in lines))


if __name__ == "__main__":
    for w in load("LICENSE"):
        print(list(w))
