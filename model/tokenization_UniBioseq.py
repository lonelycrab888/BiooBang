import os
from typing import List, Optional, Union
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        lines = f.read().splitlines()
        return [l.strip() for l in lines]
    
class UBSLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        prompt_token = "+",
        **kwargs,
    ):
        self.all_toks = load_vocab_file(vocab_file)
        self._id_to_token = dict(enumerate(self.all_toks))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_toks)}
        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            prompt_token = prompt_token,
            **kwargs,
        )

        # TODO, all the tokens are added? But they are also part of the vocab... bit strange.
        # none of them are special, but they all need special splitting.
        self.prompt_token_id = self.token_to_id(prompt_token)
        self.unique_no_split_tokens = self.all_toks
        self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        return text.split()

    def get_vocab_size(self, with_added_tokens=False):
        return len(self._id_to_token)

    def get_vocab(self):
        return {token: i for i, token in enumerate(self.all_toks)}

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls = [self.cls_token_id]
        sep = [self.eos_token_id] 
        prompt = [self.prompt_token_id]
        if token_ids_1 is None:
            if self.eos_token_id is None:
                return cls + token_ids_0
            else:
                return cls + token_ids_0 + sep
        elif self.eos_token_id is None:
            raise ValueError("Cannot tokenize multiple sequences when EOS token is not set!")
        return cls + token_ids_0 + sep + prompt + token_ids_1 + sep  # Multiple inputs always have an EOS token
        #return [token_ids_0, token_ids_1]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            if token_ids_0.count(self.eos_token_id) >=2:
                return [1 if token in self.all_special_ids or token == self.prompt_token_id else 0 for token in token_ids_0][:-1]+[0]
            else:
                return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
            
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [1] + [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, filename_prefix):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_toks))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size(with_added_tokens=False)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        return super()._add_tokens(new_tokens, special_tokens=special_tokens)
    
