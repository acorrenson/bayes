from dataclasses import dataclass
import re


def tokenize(s: str):
    token_specification = [
        ('IGNORE', r'[^[A-Za-z\d]'),
        ('WORD', r'[A-Za-z\d]+'),
        ('MISMATCH', r'.'),
    ]
    token_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    for m in re.finditer(token_regex, s):
        match m.lastgroup:
            case 'IGNORE':
                continue
            case "WORD":
                yield m.group().lower()
            case 'MISMATCH':
                raise RuntimeError(f'Unexpected token {m.group()}')
            case _:
                raise RuntimeError(f'Unknwon token kind {m.lastgroup}')
