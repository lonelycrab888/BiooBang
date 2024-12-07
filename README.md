# BiooBang-v1

This repository contains codes for BiooBang, which is an advanced biological language model designed to integrate **protein amino acid sequences** and **mRNA coding sequences (CDS)** within a unified framework. Built on a **Transformer-based prefix decoder** architecture, BiooBang leverages the principles of natural language processing to treat protein and CDS sequences as biological “languages,” enabling self-supervised learning for comprehensive training.


- [BiooBang](#BiooBang-v1)
  - [Installation](#installation)
    - [Create Environment with Conda](#create-environment-with-conda)
  - [Downstream Tasks](#downstream-tasks)
    - [Species identification](#species-identification)
      - [1. Data Preparation](#1-data-preparation-1)
      - [2. Fine-tuning](#2-fine-tuning)
      - [3. Evaluation](#3-evaluation)
    - [TE prediction](#te-prediction)
      - [1. Data Preparation](#1-data-preparation-2)
      - [2. Fine-tuning](#2-fine-tuning-1)
      - [3. Evaluation](#3-evaluation-1)
    - [MRL prediction](#mrl-prediction)
      - [1. Data Preparation](#1-data-preparation-3)
      - [2. Adaptation](#2-adaptation)
      - [3. Evaluation](#3-evaluation-2)
    - [CDS denovo generation](#cds-denovo-generation)
      - [1. Data Preparation](#1-data-preparation-3)
      - [2. Adaptation](#2-adaptation)
      - [3. Evaluation](#3-evaluation-2)
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
## downstream tasks

### Species identification

### TE prediction

### MRL prediction

### CDS denovo generation
