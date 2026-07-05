# LLMs From Scratch

A hands-on project-based exploration of Large Language Models, building components from the ground up.

## Contents

### Notebooks
| # | Notebook | Topic |
|---|----------|-------|
| 1 | `tokenization_from_scratch.ipynb` | Implementing a tokenizer from scratch |
| 2 | `byte_pair_encoding.ipynb` | Byte Pair Encoding (BPE) tokenization |
| 3 | `input_target_pairs.ipynb` | Creating input–target pairs for language modeling |
| 4 | `token_embeddings.ipynb` | Converting tokens into dense vector embeddings |
| 5 | `positional_embedding.ipynb` | Encoding token positions for sequence awareness |
| 6 | `data_preprocessing_pipeline.ipynb` | End-to-end data preprocessing for LLM training |
| 7 | `attention_mechanism.ipynb` | Scaled dot-product and multi-head attention |
| 8 | `transformer_basics.ipynb` | Building the Transformer architecture |
| 9 | `instruction_fine_tuning_intro.ipynb` | Fine-tuning LLMs on instruction datasets |
| — | `implementing_dummy_gpt_model.ipynb` | A minimal GPT-style model implementation |
| — | `Project_1.ipynb` | Capstone / miscellaneous project |

### Data
- `the-verdict.txt` — Sample text corpus
- `train.csv` / `validation.csv` / `test.csv` — Train/validation/test splits
- `sms_spam_collection/` — SMS spam classification dataset

## Getting Started

Launch any notebook with:

```bash
jupyter notebook <notebook_name>.ipynb
```

Dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `tiktoken`, `jupyter`.

## Purpose

This repository follows a bottom-up approach to understanding LLMs — starting from tokenization and building up through embeddings, attention, transformers, and fine-tuning. Each notebook is self-contained and builds on concepts introduced in earlier ones.
