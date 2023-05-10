from abc import ABC, abstractmethod, abstractproperty
# from utils.tokenizer import Tokenizer, CharTokenizer
from tokenizer import Tokenizer, WordTokenizer
from typing import Any
import torch
import numpy as np

class Batcher(ABC):
    @abstractmethod
    def __init__(self, tokenizer: Tokenizer):
        pass

    @abstractmethod
    def batch_from_tuples(self, data: tuple[Any, Any]) -> Any:
        """
        Takes list of tuples of (x, y) and converts them into a batch 
        """
        pass

    @abstractmethod
    def batch_from_Xs(self, data: Any) -> Any:
        """
        Takes list of x and converts them into a batch
        """
        pass

    def make_batch(self, data: list) -> Any:
        if isinstance(data[0], tuple):
            return self.batch_from_tuples(data)
        return self.batch_from_Xs(data)


class TensorBatcher(Batcher):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def batch_from_Xs(self, data: list[Any]) -> Any:
        """ Convert a list of tokens into a matrix with padding """
        assert isinstance(data, list), "data must be a list of strings"
        assert len(data) > 0, "Sequences must be non-empty"

        max_len = max(map(len, data))
        pad_ix = self.tokenizer.token_to_id[self.tokenizer.special_tokens['PAD']]
        matrix = np.full((len(data), max_len), np.int32(pad_ix))
        for i, seq in enumerate(data):
            matrix[i, :len(seq)] = seq
        return torch.tensor(matrix, dtype=torch.long)

    def batch_from_tuples(self, data: tuple[Any, Any]) -> Any:
        x, y = zip(*data)
        assert isinstance(x[0], list), "x must be a list of list of strings"
        assert isinstance(x[0][0], str), "x must be a list of list of strings"

        x = self.batch_from_X(x)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
    

if __name__ == "__main__":
    tokenizer = WordTokenizer(['hello', 'Hello world!', 'bla bla',  'meow meow'], min_count=1)
    batcher = TensorBatcher(tokenizer=tokenizer)
    assert batcher.tokenizer == tokenizer, "Error: tokenizer property"
    texts = ['hello', 'Hello world!', 'bla bla',  'meow meow', 'HeLLo', 'meow', 'bla', 'a a a a a']
    tokens = tokenizer(texts)
    print(tokens)
    print(batcher.make_batch(tokens))
    print("All tests passed!")