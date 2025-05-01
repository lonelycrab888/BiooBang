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
    'Xho-I': ['ccccucgag','ccgcucgag','gaucucgag'],
    'Dpn-I': ['gauc']
}


AMINO_ACID_TO_CODONS = {
    'I': ['auu', 'auc', 'aua'],
    'M': ['aug'],
    'T': ['acu', 'acc', 'aca', 'acg'],
    'N': ['aac', 'aau'],
    'K': ['aaa', 'aag'],
    'S': ['agc','agu','ucc', 'uca', 'ucg', 'uac'],
    'R': ['agg','aga','cgu', 'cgc', 'cgg', 'cga'],

    'V': ['guc', 'guu', 'gug', 'gca'],
    'A': ['gca', 'gcc', 'gcu', 'gcg'],
    'D': ['gac', 'gau'],
    'E': ['gag', 'gaa'],
    'G': ['ggc', 'ggu', 'ggg', 'gga'],

    'F': ['uuc', 'uuu'],
    'L': ['uua', 'uug', 'cua', 'cug', 'cuc', 'cuu'],
    'Y': ['uac', 'uau'],
    'C': ['ugu', 'ugc'],
    'W': ['ugg'],

    'P': ['ccu', 'ccc', 'cca', 'ccg'],
    'H': ['cau', 'cac'],
    'Q': ['caa', 'cag'],

    '*': ['uaa', 'uag', 'uga'],
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




class CodonLogitsProcessor(LogitsProcessor):
    def __init__(self, amino_acid_seq, tokenizer, input_length):

        self.amino_acids = amino_acid_seq
        self.codon_table = AMINO_ACID_TO_CODONS
        self.tokenizer = tokenizer
        self.input_length = input_length
        
        self.id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        
    def __call__(self, input_ids, scores):
        current_step = input_ids.shape[-1]-3  #ignore special tokens
        cds_position = current_step - self.input_length
        
        if cds_position < 0:
            return scores
        
        codon_idx = cds_position // 3
        pos_in_codon = cds_position % 3
        
        if codon_idx >= len(self.amino_acids)+3:
            scores[:, :] = -float('inf')  
            return scores
        
        if codon_idx < len(self.amino_acids):
            current_aa = self.amino_acids[codon_idx]
        else:
            current_aa = '*'
        
        possible_codons = self.codon_table.get(current_aa, [])

        for batch_idx in range(input_ids.shape[0]):
            seq = input_ids[batch_idx]
            
            generated_cds = seq[self.input_length+3:]
            
            codon_start = codon_idx * 3
            current_codon_chars = [
                self.id_to_token.get(tok.item(), '') 
                for tok in generated_cds[codon_start:codon_start+pos_in_codon]
            ]
            
            allowed_chars = []
            if pos_in_codon == 0:
                allowed_chars = {codon[0] for codon in possible_codons}
            elif pos_in_codon == 1:
                valid_codons = [c for c in possible_codons 
                               if len(c) > 0 and c[0] == current_codon_chars[0]]
                allowed_chars = {c[1] for c in valid_codons}
            elif pos_in_codon == 2:
                valid_codons = [c for c in possible_codons 
                               if len(c) >= 2 and 
                               c[0] == current_codon_chars[0] and 
                               c[1] == current_codon_chars[1]]
                allowed_chars = {c[2] for c in valid_codons}
            
            allowed_ids = []
            for char in allowed_chars:
                token_id = self.tokenizer.convert_tokens_to_ids(char)
                if token_id != self.tokenizer.unk_token_id:
                    allowed_ids.append(token_id)
            
            if not allowed_ids:
                scores[batch_idx, :] = -float('inf')
                continue
                
            mask = torch.ones_like(scores[batch_idx]) * -float('inf')
            for allowed_id in allowed_ids:
                mask[allowed_id] = 0
            scores[batch_idx] += mask
            
        return scores