import random


def generate_word(label: bool):
    w = ""
    M = random.randint(1, 10)
    for _ in range(M):
        if random.randint(0, 10) < 7:
            w += 'a' if label else 'b'
        else:
            w += 'b' if label else 'a'
    return w


def generate_sentence(label: bool):
    N = random.randint(1, 15)
    for _ in range(N):
        yield generate_word(label)


def generate_data(n: int, label: bool):
    for _ in range(n):
        yield list(generate_sentence(label))
