"""util.py"""

from typing import List

def collect_first_k_words(words: list, k):
    """return the first k words per ingredient in any list of ingredients"""
    yield from words.split(" ")[:k]

def split(a: List, n: int):
    """divide training data into batches for saving/live streaming during training"""
    q = len(a) // n
    assert q > 0
    k, m = divmod(len(a), q)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(q))

def split_iter(a: List, n: int):
    """divide training data into batches for saving/live streaming during training"""
    q = len(a) // n - 1
    assert q > 0
    for i in range(q):
        yield a[i*n:(i+1)*n]
