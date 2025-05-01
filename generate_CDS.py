from model.tokenization_UniBioseq import UBSLMTokenizer
from model.modeling_UniBioseq import UniBioseqForCausalLM
from model.UBL_utils import ForbidSequenceLogitsProcessor, CodonLogitsProcessor
from transformers.generation.logits_process import LogitsProcessorList
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse

from tqdm import tqdm

def read_fa(fasta_path):
    fasta_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


def write_dict_to_fasta(fasta_dict, output_file):
    records = []
    for seq_id, sequence in fasta_dict.items():
        record = SeqRecord(Seq(sequence), id=seq_id, description="")
        records.append(record)
    SeqIO.write(records, output_file, "fasta")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='input protein fasta path')
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--num_beams', type=int, default=10, help='beamsearch width')
    parser.add_argument('--enzyme_cleavage_sites', type=str, default=None, help='Enzyme cleavage sites that need to be excluded can be provided as a list of sequences.')
    args = parser.parse_args() 
    fasta_path = args.input_path
    save_path = args.save_path
    num_beams = args.num_beams 
    print(fasta_path)
    excluding_sites = args.enzyme_cleavage_sites.split(',') if args.enzyme_cleavage_sites else None
    print("excluding_sites: ", excluding_sites)

    print("load pretrained model!")
    # Load the tokenizer and model
    tokenizer = UBSLMTokenizer.from_pretrained("pretrained-model/BiooBang-generationCDS")
    model = UniBioseqForCausalLM.from_pretrained("pretrained-model/BiooBang-generationCDS", device_map='auto')

    # Load the prompt data
    input_protein_dict = read_fa(fasta_path)

    print("start generation!")
    output_cds = {}
    for name, input_protein in tqdm(input_protein_dict.items()):
        if input_protein[0]=='M':
            M_flag = True 
        else:
            M_flag =False
            input_protein = 'M'+input_protein
        input_ids = torch.tensor([tokenizer.encode(input_protein)+[36]]).to(model.device)
        max_length = 4*len(input_protein)+6

        # define the logits processor
        logits_processor = LogitsProcessorList()
        logits_processor.append(CodonLogitsProcessor(input_protein, tokenizer, len(input_protein)))
        logits_processor.append(ForbidSequenceLogitsProcessor(tokenizer, excluding_sites))

        # generate the sequence
        result = model.generate(input_ids, max_length = max_length, num_beams = num_beams, logits_processor=logits_processor, low_memory=True, num_return_sequences=1)
        
        # decode the result
        result_tok = tokenizer.decode(result[0][len(input_protein)+3:].tolist()).replace(" ","").upper()
        if M_flag == False:
            result_tok = result_tok[3:]
        output_cds[name] = result_tok

    print(f"finish!")
    write_dict_to_fasta(output_cds, save_path)
    print(f"save results to {save_path}")
    
if __name__ == '__main__':
    main()
    