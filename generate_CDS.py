from model.tokenization_UniBioseq import UBSLMTokenizer
from model.modeling_UniBioseq import UniBioseqForCausalLM
from model.UBL_utils import new_logits_processor, forbid_aa
from transformers.generation.logits_process import LogitsProcessorList
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse

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
    parser.add_argument('--input_path', type=str, default='experiment_data/input.fasta', help='input protein fasta path')
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--num_beams', type=int, default=10, help='beamsearch width')
    args = parser.parse_args() 
    fasta_path = args.input_path
    save_path = args.save_path
    num_beams = args.num_beams


    print("load pretrained model!")
    tokenizer = UBSLMTokenizer.from_pretrained("pretrained-model/BiooBang-generationCDS")
    model = UniBioseqForCausalLM.from_pretrained("pretrained-model/BiooBang-generationCDS", device_map='auto')
    logits_processor = LogitsProcessorList()
    logits_processor.append(new_logits_processor(forbid_aa()))
    input_protein_dict = read_fa(fasta_path)
    print(input_protein_dict)
    output_cds = {}
    print("start generation!")
    for name, input_protein in input_protein_dict.items():
        input_ids = torch.tensor([tokenizer.encode(input_protein)+[36]]).to(model.device)
        max_length = 4*len(input_protein)+6
        result = model.generate(input_ids, max_length = max_length, num_beams = num_beams, logits_processor=logits_processor, low_memory=True, num_return_sequences=1)
        result_tok_list = []
        result_tok = tokenizer.decode(result[0][len(input_protein)+3:].tolist()).replace(" ","").upper()
        output_cds[name] = result_tok
    print(f"finish!")
    write_dict_to_fasta(output_cds, save_path)
    print(f"save results to {save_path}")

if __name__ == '__main__':
    main()
