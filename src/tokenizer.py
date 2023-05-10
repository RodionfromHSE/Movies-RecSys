from abc import ABC, abstractmethod
from typing import Any
from collections import Counter, defaultdict
from nltk import WordPunctTokenizer as WPT
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')

class Tokenizer(ABC):
    def __init__(self, texts, special_tokens=None, min_count=10):
        if special_tokens is None:
            special_tokens = {'UNK': 'UNK', 'PAD': 'PAD'}
        assert isinstance(special_tokens, dict), 'special_tokens must be a dict'
        assert 'UNK' in special_tokens, 'special_tokens must have UNK'
        assert 'PAD' in special_tokens, 'special_tokens must have PAD'
        self.special_tokens = special_tokens
        self.__init_token_to_id(texts, min_count)

    def __init_token_to_id(self, texts, min_count):
        cnt = Counter()
        # print(texts)
        for text in texts:
            cnt.update(self.tokenize(text))
        tokens = [token for token, count in cnt.items() if count >= min_count]
        # print(f'Number of tokens: {len(tokens)}')
        # print(f'Tokens: {tokens[:10]}')
        tokens = list(self.special_tokens.values()) + tokens
        token_to_id = {token: i for i, token in enumerate(tokens)}
        self.token_to_id = defaultdict(lambda: token_to_id[self.special_tokens['UNK']], token_to_id)

    @abstractmethod
    def tokenize(self, text):
        pass

    def text_to_ids(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id[token] for token in tokens]

    def __call__(self, data):
        if isinstance(data, str):
            return self.text_to_ids(data)
        return [self.text_to_ids(text) for text in data]
    

class WordTokenizer(Tokenizer):
    def __init__(self, texts, special_tokens=None, min_count=10):
        self.tokenizer = WPT()
        super().__init__(texts, special_tokens, min_count)

    def tokenize(self, text):
        raw_tokens = self.tokenizer.tokenize(text.lower())
        return [token for token in raw_tokens if token not in string.punctuation and token not in stopwords.words('english')]
    

if __name__ == '__main__':
    tk = WordTokenizer(['hello', 'hello world!'], min_count=1)
    assert set(tk.token_to_id.values()) == {0, 1, 2, 3}, f"Wrong token_to_id values: {tk.token_to_id}"
    assert set(tk.token_to_id.keys()) == {'UNK', 'PAD', 'hello', 'world'}, f"Wrong token_to_id keys: {set(tk.token_to_id.keys())}"
    assert tk('hello world!') == [2, 3], tk('hello world!')
    print('Test passed!')
