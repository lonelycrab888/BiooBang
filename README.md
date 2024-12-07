# BiooBang-v1

This repository contains codes for BiooBang, which is an advanced biological language model designed to integrate **protein amino acid sequences** and **mRNA coding sequences (CDS)** within a unified framework. Built on a **Transformer-based prefix decoder** architecture, BiooBang leverages the principles of natural language processing to treat protein and CDS sequences as biological “languages”, enabling self-supervised learning for comprehensive training.


- [BiooBang](#BiooBang-v1)
  - [Installation](#installation)
    - [Create Environment with Conda](#create-environment-with-conda)
  - [Get embeddings](#get-embeddings)
  - [CDS denovo generation](#cds-denovo-generation)
    - [1. Data Preparation](#1-data-preparation-2)
    - [2. Evaluation](#2-evaluation-2)
  - [Baselines](#baselines)
  - [Citation](#citation)

## Installation

### Create Environment with Conda
First, download the repository and create the environment.
```bash
git clone https://github.com/lonelycrab888/BiooBang.git
cd ./BiooBang
conda env create -f environment.yml
```
Then, activate the "BiooBang" environment.

```bash
conda activate BiooBang
```

## Get Embeddings
```python
import torch
from model.modeling_UniBioseq import UniBioseqForEmbedding
from model.tokenization_UniBioseq import UBSLMTokenizer
model_file = "./load_files/BiooBang_FM"
# ========== Set device
device = "cuda:0"

# ========== Prepare Data
data = [
    ("Protein", "MASSDKQTSPKPPPSPSPLRNSKFCQSNMRILIS"),
    ("RNA", "ATGGCGTCTAGTGATAAACAAACAAGCCCAAAGCCTCCTCCTTCACCGTCTCCTCTCCGTAATT")
]

# ========== BiooBang Model
model = UniBioseqForEmbedding.from_pretrained(moded_file)
model.eval()
model.to(device)
# ========== get Embeddings
embeddings = {}
for name,iput_seq in data:
    input_ids = tokenizer.encode(input_seq)
    with torch.no_grad():
        embeddings[name] = model(input_ids).logits.tolist()
```

## CDS denovo generation
