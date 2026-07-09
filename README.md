# LLMs From Scratch

A hands-on project-based exploration of Large Language Models, building components from the ground up.

## Contents

### Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `tokenization_from_scratch.ipynb` | Implementing a tokenizer from scratch |
| 2 | `byte_pair_encoding.ipynb` | Byte Pair Encoding (BPE) tokenization |
| 3 | `input_target_pairs.ipynb` | Creating input-target pairs for language modeling |
| 4 | `token_embeddings.ipynb` | Converting tokens into dense vector embeddings |
| 5 | `positional_embedding.ipynb` | Encoding token positions for sequence awareness |
| 6 | `data_preprocessing_pipeline.ipynb` | End-to-end data preprocessing for LLM training |
| 7 | `attention_mechanism.ipynb` | Scaled dot-product and multi-head attention |
| 8 | `transformer_basics.ipynb` | Building the Transformer architecture |
| 9 | `instruction_fine_tuning_intro.ipynb` | Fine-tuning LLMs on instruction datasets |
| 10 | `instruction_fine_tuning_dataloader_training.ipynb` | DataLoaders, label masking, and full training loop for instruction fine-tuning |
| 11 | `llm_fine_tuning_training_loop_and_ollama_eval.ipynb` | Manual PyTorch fine-tuning loop, generation, and Ollama-based evaluation |
| - | `implementing_dummy_gpt_model.ipynb` | A minimal GPT-style model implementation |
| - | `Project_1.ipynb` | Capstone / miscellaneous project |

### Final Project Files

- `FINAL_REPORT.md` - Consolidated project report covering the full LLM learning path, training result, limitations, and future improvements.

### Data

- `the-verdict.txt` - Sample text corpus
- `train.csv` / `validation.csv` / `test.csv` - Train/validation/test splits
- `sms_spam_collection/` - SMS spam classification dataset

## Getting Started

Launch any notebook with:

```bash
jupyter notebook <notebook_name>.ipynb
```

Common dependencies:

```bash
pip install torch numpy pandas matplotlib tiktoken jupyter
```

Notebook 11 also uses:

```bash
pip install transformers datasets tqdm requests
```

Notebook 11 downloads `distilgpt2` from Hugging Face the first time it runs, so internet access is needed for the initial model download.

### Ollama Evaluation Setup

The fine-tuning part of notebook 11 does not require Ollama. Ollama is only needed for the final LLM-as-judge evaluation section.

Install Ollama, then pull a local judge model:

```bash
ollama pull llama3.1:8b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

If `ollama serve` reports that port `11434` is already in use, Ollama is already running in the background and you can continue.

If you use another installed Ollama model, such as `gemma3:latest`, update this variable in notebook 11:

```python
OLLAMA_JUDGE_MODEL = "gemma3:latest"
```

## Purpose

This repository follows a bottom-up approach to understanding LLMs - starting from tokenization and building up through embeddings, attention, transformers, and fine-tuning. Each notebook is self-contained and builds on concepts introduced in earlier ones.
