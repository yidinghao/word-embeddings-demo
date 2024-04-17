"""
Author: Sophie Hao
Date: April 17, 2024

This module contains the Embeddings class, which will help us work with
word embeddings more easily.
"""
from typing import Iterable, Optional

import numpy as np


class Embeddings:
    """
    This class represents a container that holds a collection of words
    and their corresponding word embeddings.
    """

    def __init__(self, words: Iterable[str], vectors: np.ndarray):
        """
        Initializes an Embeddings object directly from a list of words
        and their embeddings.

        :param words: A list of words
        :param vectors: A 2D array of shape (len(words), embedding_size)
            where for each i, vectors[i] is the embedding for words[i]
        """
        self.words = list(words)
        self.indices = {w: i for i, w in enumerate(words)}
        self.vectors = vectors

    def __len__(self):
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def __getitem__(self, words: Iterable[str]) -> np.ndarray:
        """
        Retrieves embeddings for a list of words.

        :param words: A list of words
        :return: A 2D array of shape (len(words), embedding_size) where
            for each i, the ith row is the embedding for words[i]
        """
        if isinstance(words, str):
            words = [words]
        return self.vectors[[self.indices[w] for w in words]]

    @classmethod
    def from_file(cls, filename: str,
                  vocab_size: Optional[int] = None,
                  alpha_only: bool = True) -> "Embeddings":
        """
        Initializes an Embeddings object from a .txt file containing
        word embeddings in GloVe or word2vec format.

        :param filename: The name of the file containing the embeddings
        :param vocab_size: Only load this many embeddings (if specified)
        :param alpha_only: If True, skip words with non-alphabetic
            characters

        :return: An Embeddings object containing the loaded embeddings
        """
        with open(filename, "r") as f:
            all_lines = [line.strip().split(" ", 1) for line in f]

        # Skip first line if the .txt file is in word2vec format
        if all(c.isnumeric() for c in all_lines[0]):
            all_lines.pop(0)

        # Parse lines and filter by vocab_size and alpha_only
        words, vecs = zip(*[line for line in all_lines[:vocab_size]
                            if (not alpha_only) or line[0].isalpha()])

        return cls(words, np.array([np.fromstring(v, sep=" ") for v in vecs]))
