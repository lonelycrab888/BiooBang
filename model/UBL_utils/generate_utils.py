from typing import List
import torch
from transformers.generation.logits_process import LogitsProcessor


class new_logits_processor(LogitsProcessor):
    def __init__(self, forbid_token_id_list: List[int] = None):
        self.forbid_token_id_list = forbid_token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for id_ in self.forbid_token_id_list:
            scores[:, id_] = -float('inf')
        return scores
    
def forbid_aa():
    forbid_words = [i for i in range(31)]
    forbid_words.extend([i for i in range(35,41)])
    return forbid_words

