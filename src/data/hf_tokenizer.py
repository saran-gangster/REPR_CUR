from typing import List

class HFTokenizer:
    def __init__(self, tokenizer_json_path: str):
        try:
            from tokenizers import Tokenizer
        except Exception as e:
            raise ImportError("Please `pip install tokenizers` to use the HF BPE tokenizer.") from e
        self.tk = Tokenizer.from_file(tokenizer_json_path)
        self.vocab_size = self.tk.get_vocab_size()

    def encode(self, s: str) -> List[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: List[int]) -> str:
        try:
            return self.tk.decode(ids)
        except Exception:
            return ""