from abc import ABC, abstractmethod
from typing import Any
from collections import defaultdict
from nltk import WordPunctTokenizer as WPT
from nltk.corpus import stopwords
import string
from collections import defaultdict
from typing import Any

class Tokenizer(ABC):
    @abstractmethod
    def __init__(self, tokens: list, special_tokens: dict) -> Any:
        self.__init_special_tokens(special_tokens)
        self.__init_token_utils(tokens)
    
    @abstractmethod
    def tokenize(self, text: str) -> list:
        pass

    def decode(self, token_ids: list) -> list:
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        return tokens

    # eos, bos, unk, pad
    def __init_special_tokens(self, special_tokens: dict):
        assert isinstance(special_tokens, dict), "special_tokens must be a dictionary"
        assert ['bos', 'eos', 'unk', 'pad'] == list(special_tokens.keys()), "special_tokens must contain bos, eos, unk, pad"
        self.special_tokens = special_tokens

    def __init_token_utils(self, tokens: list):
        """	
        init token_to_id and id_to_token
        """
        assert isinstance(tokens, list), "tokens must be a list of strings"
        assert len(tokens) > 0, "tokens must be non-empty"
        assert len(set(tokens)) == len(tokens), "tokens must be unique"
        # add special tokens to tokens set
        tokens = set(tokens).union(set(self.special_tokens.values()))
        self.len = len(tokens)
        token_to_id = {token: i for i, token in enumerate(tokens)}

        def factory():
            print('Warning: unknown token')
            return token_to_id[self.special_tokens['unk']]
        self.token_to_id = defaultdict(factory, token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def __call_single(self, text: str) -> list:
        tokens = self.tokenize(text)
        token_ids = [self.token_to_id[token] for token in tokens]
        return token_ids

    def __call__(self, text: Any) -> Any:
        """	tokenize text or list of texts, i.e return list of token_ids """
        if isinstance(text, str):
            return self.__call_single(text)
        return [self.__call_single(t) for t in text]
    
    def __len__(self) -> int:
        """	returns number of tokens """
        return self.len
    

class WordTokenizer(Tokenizer):
    def __init__(self, tokens, special_tokens=None):
        if special_tokens is None:
            special_tokens = {'bos': 'bos', 'eos': 'eos', 'unk': 'unk', 'pad': 'pad'}
        super().__init__(tokens, special_tokens)
        # ignore punctuation and stopwords
        self.tokenizer = WPT()

    def tokenize(self, text):
        raw_tokens = self.tokenizer.tokenize(text.lower())
        return [token for token in raw_tokens if token not in string.punctuation and token not in stopwords.words('english')]

if __name__ == '__main__':
    tokens = ['hello', 'world']
    tk = WordTokenizer(tokens)
    inp = 'hello world! bonjour?'
    tokenized = tk(inp)
    assert len(tk) == 6, f"Wrong number of tokens: {len(tk)}"
    assert tk.decode(tokenized) == ['hello', 'world', tk.special_tokens['unk']], f"Wrong decoding: {tk.decode(tokenized)}"
    assert tokenized == [tk.token_to_id['hello'], tk.token_to_id['world'], tk.token_to_id['unk']], f"Wrong tokenization: {tokenized}"
    print('Test passed!')
