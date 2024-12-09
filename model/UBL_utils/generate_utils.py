from typing import List
import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers import LogitsProcessor, LogitsProcessorList
import torch
from typing import List

sites = {
    'BamH-I': ['cgggaucc','cgcggaucc'],
    'EcoR-I': ['gacgaauuc','ccggaauuc'],
    'Hind-III': ['ccaagcuu','cccaagcuu'],
    'Kpn-I': ['gggguacc','cgggguacc'],
    'Nco-I': ['cccaugg','caugccaugg'],
    'Nde-I': ['cgccauaug','ggguuucauaug','ggaauuccauaug','gggaauuccauaug'],
    'Nhe-I': ['cggcuagc','cuagcuagc'],
    'Not-I': ['uugcggccgc','auuugcggccgc','aaauaugcggccgc','auaagaaugcggccgc'],
    'Sac-I': ['cgagcuc'],
    'Sal-I': ['gcgucgac','acgcgucgac'],
    'Sma-I': ['cccccggg','ucccccggg'],
    'Xba-I': ['ugcucuaga','cuagucuaga'],
    'Xho-I': ['ccccucgag','ccgcucgag','gaucucgag']
}

def forbid_aa():
    forbid_words = [i for i in range(31)]
    forbid_words.extend([i for i in range(35,41)])
    return forbid_words

def forbid_seq(sequences, tokenizer):
    forbid_token_ids = []
    for i in sequences:
        if i in list(sites.keys()):
            for j in sites[i]:
                forbid_token_ids.append(tokenizer.encode(j)[1:-1])
        else:
            sequence = i.replace('T', 'u').lower()
            forbid_token_ids.append(tokenizer.encode(sequence)[1:-1])
    return forbid_token_ids

class ForbidSequenceLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, forbid_sequences: List[str] = None):
        if forbid_sequences is not None:
            self.forbid_sequences = forbid_seq(forbid_sequences, tokenizer)
        else:
            self.forbid_sequences = None
        
        self.forbid_tokens = forbid_aa()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        for id_ in self.forbid_tokens:
            scores[:, id_] = -float('inf')
        if self.forbid_sequences is None:
            return scores
        else:
            for seq in self.forbid_sequences:
                seq_len = len(seq)
                
                if input_ids.shape[1] < seq_len - 1:
                    continue
                
                for batch_idx in range(input_ids.shape[0]):
                    if input_ids[batch_idx, -seq_len + 1:].tolist() == seq[:-1]:
                        scores[batch_idx, seq[-1]] = -float('inf')
            return scores
