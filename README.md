<div align="center">

# BiooBang 1.0

[![biorxiv](https://img.shields.io/badge/bioRxiv-preprint-yellow.svg)](https://www.biorxiv.org/content/10.1101/2024.10.24.620004v1.full.pdf)
[![HuggingFace Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-red)](https://huggingface.co/collections/lonelycrab88/bioobang-10-68185e7a37bd54dd54707b12)
[![Zenodo](https://img.shields.io/badge/zenodo-Datasets-green)](https://doi.org/10.5281/zenodo.14648464)
![license](https://img.shields.io/badge/License-CC--BY--NC%204.0-blue.svg?labelColor=gray)


</div>


This repository contains codes for BiooBang, which is an advanced biological language model designed to integrate **protein amino acid sequences** and **mRNA coding sequences (CDS)** within a unified framework. Built on a **Transformer-based prefix decoder** architecture, BiooBang leverages the principles of natural language processing to treat protein and CDS sequences as biological “languages”, enabling self-supervised learning for comprehensive training.


- [BiooBang](#BiooBang-v1)
  - [Installation](#installation)
    - [Create Environment with Conda](#create-environment-with-conda)
    - [Download Pre-trained Models](#download-pre-trained-models)
  - [Quick start](#quick-start)
  - [Protein property prediction](#protein-property-prediction)
  - [CDS denovo generation](#cds-denovo-generation)
  - [Citing this work](#citing-this-work)
  - [Contacts](#Contacts)
      

## Installation

### Create Environment with Conda
First, download the repository and create the environment.
```bash
git clone https://github.com/lonelycrab888/BiooBang.git
cd ./BiooBang
conda env create -f environment.yml
### We test with cu118
### Notice to check your cuda version
```
Then, activate the "BiooBang" environment.

```bash
conda activate BiooBang
```

Typical install time on a “normal" desktop computer: “Approximately 5-10 minutes, assuming a stable internet connection and pre-installed Python (version 3.9 or later).“

### Download Pre-trained Models
The model can be easily loaded using the .from_pretrained("lonelycrab88/BiooBang-1.0") method.
Another way to use the model is by downloading the weights locally from [Huggingface](https://huggingface.co/lonelycrab88/BiooBang-1.0/tree/main).

## Quick Start

```python
import torch
from model.modeling_UniBioseq import UniBioseqForEmbedding
from model.tokenization_UniBioseq import UBSLMTokenizer
# ========== Set device
device = "cuda:0"

# ========== Prepare Data
data = [
    ("Protein", "MASSDKQTSPKPPPSPSPLRNSKFCQSNMRILIS"),
    ("RNA", "ATGGCGTCTAGTGATAAACAAACAAGCCCAAAGCCTCCTCCTTCACCGTCTCCTCTCCGTAATT")
]

# ========== BiooBang Model
model = UniBioseqForEmbedding.from_pretrained("lonelycrab88/BiooBang-1.0")
tokenizer = UBSLMTokenizer.from_pretrained("lonelycrab88/BiooBang-1.0")
model.eval()
model.to(device)
# ========== get Embeddings
embeddings = {}
hidden_states = {}
for name,input_seq in data:
    input_ids = tokenizer(input_seq, return_tensors="pt")['input_ids'].to(device)
    with torch.no_grad():
        # get sequence embedding 
        embeddings[name] = model(input_ids).logits
        # get last hidden states (token embeddings)
        hidden_states[name] = model(input_ids).hidden_states[:,1:-1,:]

# ========== generate CDS
from transformers.generation.logits_process import LogitsProcessorList
from model.UBL_utils import CodonLogitsProcessor
from model.modeling_UniBioseq import UniBioseqForCausalLM
tokenizer = UBSLMTokenizer.from_pretrained("lonelycrab88/BiooBang-1.0")
model = UniBioseqForCausalLM.from_pretrained("lonelycrab88/BiooBang-1.0", device_map='auto')

protein_prompt = "MASSDKQTSPKPPPSPSPLRNSKFCQSNMRILIS"
input_ids = torch.tensor([tokenizer.encode(input_protein)+[36]]).to(model.device)
max_length = 4*len(input_protein)+6

logits_processor = LogitsProcessorList()
logits_processor.append(CodonLogitsProcessor(input_protein, tokenizer, len(input_protein)))
result = model.generate(input_ids, max_length = max_length, num_beams = 10, logits_processor=logits_processor, low_memory=True, num_return_sequences=1)
result_CDS_tok = tokenizer.decode(result[0][len(input_protein)+3:].tolist()).replace(" ","").upper()

```

The expected output dimension of the embedding vector is 1280 dimensions. We offer two options, namely hidden_states and logits. The hidden_states contain the embedding vectors of each token, while the logits represent the sentence embeddings.

## Protein Property Prediction

We provide a notebook [Example Notebook](validation_benchmark/protein_benchmark_test.ipynb) for protein property prediction, including tasks such as SolP, FoldP, LocP, and SSP.
We fine-tuned the base model using the frozen parameter method (training only the prediction head). 

To use this notebook, you can download the data and our model weights from [Zenodo](https://zenodo.org/records/14648464). 

You can find our test data in the `training_data` folder, and the fine-tuned model weights in the `finetuned_model` directory.
We provide model checkpoints trained with three different random seeds.
The [Zenodo](https://zenodo.org/records/14648464) only provides the weights of the prediction heads. 
When using the notebook, you can use the [merge_weights](validation_benchmark/merge_weights.py) script to recreate the complete weight file. When putting the created weight file into the `model_file` folder, please rename it to `pytorch_model.bin`, and you’ll be able to easily run our test notebook.

### How to use [merge_weights](validation_benchmark/merge_weights.py)


usage: python merge_params.py [-h] --head_weights_path HEADS_WEIGHTS --output_path OUTPUT_PATH

Merge model weights script

options:

  -h, --help            show this help message and exit
  
  --head_weights_path HEADS_WEIGHTS : the path to the prediction head weights file
			
  --output_path OUTPUT_PATH : the path to save the merged weights file


## CDS Denovo Generation


```bash
python generate_CDS.py \
      --input_path='your_path' \
      --save_path='your_save_path' \
      --num_beams=10 \
      --enzyme_cleavage_sites='BamH-I,cggcuaGc'
```

Default:

  beam_width: 10

  enzyme_cleavage_sites: None 

We provide the following enzyme cleavage sites for selection to be excluded:

```bash
‘BamH-I’, ‘EcoR-I’, ‘Hind-III’, ‘Kpn-I’, 
‘Nco-I’, ‘Nde-I’, ‘Nhe-I’, ‘Not-I’, ‘Dpn-I’, 
‘Sac-I’, ‘Sal-I’, ‘Sma-I’, ‘Xba-I’, ‘Xho-I’.
```

**Precautions**: If your sequence is too long or computational resources are insufficient, we recommend performing truncated generation for the target protein and stitching the segments together at the end.

Two proteins (GFP and mCherry) are used as examples in this demo script. You can set the input_path to `./experiment_data/input.fasta` to reproduce the results in our paper.

For our example proteins (beam_width=50), we used 8 * NVIDIA GeForce RTX 3090(24G), and the running time was approximately 3 minutes per protein. The length of the protein is positively correlated with the duration of running inference. 

**More versatile fine-tuned models for a broader range of cell lines are on the verge of being unleashed—prepare for the revolution!**


## Citing this Work

Please cite our paper:

```bibtex
@article {Zhao2024.10.24.620004,
	author = {Zhao, Heng-Rui and Cheng, Meng-Ting and Zhu, Jinhua and Wang, Hao and Yang, Xiang-Rui and Wang, Bo and Sun, Yuan-Xin and Fang, Ming-Hao and Chen, Enhong and Li, Houqiang and Han, Shu-Jing and Chen, Yuxing and Zhou, Cong-Zhao},
	title = {Integration of protein and coding sequences enables mutual augmentation of the language model},
	elocation-id = {2024.10.24.620004},
	year = {2024},
	doi = {10.1101/2024.10.24.620004},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/10/29/2024.10.24.620004},
	eprint = {https://www.biorxiv.org/content/early/2024/10/29/2024.10.24.620004.full.pdf},
	journal = {bioRxiv}
}
```


## Contacts

If you’re interested in other cell lines and open to collaboration, please don’t hesitate to contact us! 

We are honored to help you if you have any questions. Please feel free to open an issue or contact us directly. Hope our code helps and look forward to your citations.

[zcz@ustc.edu.cn]|[zhr123456@mail.ustc.edu.cn].
