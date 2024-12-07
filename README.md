# BiooBang-v1

This repository contains codes for BiooBang, which is an advanced biological language model designed to integrate **protein amino acid sequences** and **mRNA coding sequences (CDS)** within a unified framework. Built on a **Transformer-based prefix decoder** architecture, BiooBang leverages the principles of natural language processing to treat protein and CDS sequences as biological “languages”, enabling self-supervised learning for comprehensive training.


- [BiooBang](#BiooBang-v1)
  - [Installation](#installation)
    - [Create Environment with Conda](#create-environment-with-conda)
    - [Download Pre-trained Models](#download-pretrained-models)
  - [Get embeddings](#get-embeddings)
  - [CDS denovo generation](#cds-denovo-generation)
      
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
### Download Pre-trained Models
Our pre-trained model could be downloaded from [Google Drive](https://drive.google.com/drive/folders/1vw8UOTkT3bbAdrdYwoFiDiNymlUYA-uu) and place the `pytorch_model.bin` files in the `./pretrained_model/../` folder.

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
model = UniBioseqForEmbedding.from_pretrained(model_file)
tokenizer = UBSLMTokenizer.from_pretrained(model_file)
model.eval()
model.to(device)
# ========== get Embeddings
embeddings = {}
for name,input_seq in data:
    input_ids = tokenizer(input_seq, return_tensors="pt")['input_ids'].to(device)
    with torch.no_grad():
        #embeddings[name] = model(input_ids).logits
        embeddings[name] = model(input_ids).hidden_states[:,1:-1,:]
```

## CDS denovo generation


```bash
python generate_CDS.py --input_path your_path --save_path your_save_path
```

Default:
  beam_width = 50
