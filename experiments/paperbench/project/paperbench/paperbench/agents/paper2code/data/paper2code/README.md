## Paper2Code Benchmark

### Overview
The **Paper2Code Benchmark** is designed to evaluate the ability to reproduce methods and experiments described in scientific papers.

We collected **90 papers** from **ICML 2024**, **NeurIPS 2024**, and **ICLR 2024**, selecting only those with publicly available GitHub repositories.  
To ensure manageable complexity, we filtered for repositories with fewer than **70,000 tokens**.  
Using a model-based evaluation, we selected the **top 30 papers** from each conference based on repository quality.

A full list of the benchmark papers is provided in `dataset_info.json`.

For more details, refer to Section 4.1 "Paper2Code Benchmark" of the [paper](https://arxiv.org/abs/2504.17192).

To use the dataset on Hugging Face, see the [Paper2Code dataset page](https://huggingface.co/datasets/iaminju/paper2code).

### How to Use
- Unzip the `paper2code_data.zip` file:
```bash
unzip paper2code_data.zip
```
- If you also need the JSON files paired with the original PDFs, please download `paper2code_full_data.zip` from [this link](https://drive.google.com/file/d/1OTO6nk8s7Q8FzRm2FeyqFLI06l1NCujc/view?usp=drive_link).

### Data Structure
Each conference folder is organized as follows:
- `[PAPER].json` — Parsed version of the paper
- `[PAPER]_cleaned.json` — Preprocessed version for PaperCoder
- `pdfs/[PAPER].pdf` — Original paper PDF (only available in `paper2code_full_data.zip`)

```bash
├── iclr2024 
├── icml2024
└── nips2024
    ├── adaptive-randomized-smoothing.json
    ├── adaptive-randomized-smoothing_cleaned.json
    ├── ... 
    ├── YOLA.json
    ├── YOLA_cleaned.json
    └── pdfs  # only available in `paper2code_full_data.zip`
        ├── adaptive-randomized-smoothing.pdf
        ├── ... 
        └── YOLA.pdf
```

### License and Usage
This dataset is a parsed version of publicly available papers from ICML, ICLR, and NeurIPS. 
The original papers are under copyright by the respective conferences or authors. 
