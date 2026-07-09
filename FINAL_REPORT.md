# Final Report: LLMs From Scratch

## Project Overview

This repository is a learning-focused walkthrough of Large Language Model fundamentals. It starts with tokenization, moves through embeddings and attention, then reaches instruction fine-tuning and evaluation concepts.

## What Is Present

```text
LLMs/
  1. tokenization_from_scratch.ipynb
  2. byte_pair_encoding.ipynb
  3. input_target_pairs.ipynb
  4. token_embeddings.ipynb
  5. positional_embedding.ipynb
  6. data_preprocessing_pipeline.ipynb
  7.attention_mechanism.ipynb
  8. transformer_basics.ipynb
  9. instruction_fine_tuning_intro.ipynb
  10. instruction_fine_tuning_dataloader_training.ipynb
  11. llm_fine_tuning_training_loop_and_ollama_eval.ipynb
  implementing_dummy_gpt_model.ipynb
  Project_1.ipynb
  sms_spam_collection/
  train.csv
  validation.csv
  test.csv
  the-verdict.txt
  README.md
  FINAL_REPORT.md
```

## Learning Path

### 1. Tokenization

The first notebooks explain how raw text is converted into tokens. This includes basic token splitting, token IDs, encoding, decoding, and why tokenization is required before text can be passed into a neural network.

### 2. Byte Pair Encoding

The BPE notebook introduces subword tokenization. This is important because modern LLMs need to represent common words efficiently while still handling rare or unseen words through smaller pieces.

### 3. Input and Target Pairs

The language modeling notebooks show how text is converted into input-target examples. This is the core training setup for next-token prediction.

### 4. Embeddings and Positional Information

The embedding notebooks explain how token IDs become dense vectors and how positional embeddings give the model information about token order.

### 5. Attention and Transformer Basics

The attention and transformer notebooks explain the core mechanism behind modern LLMs: tokens attend to other tokens, combine context, and pass through Transformer blocks.

### 6. Instruction Fine-Tuning

The instruction fine-tuning notebooks show how a pretrained model can be adapted to prompt-response style data. They cover prompt formatting, batching, masking labels, training loops, validation loss, and evaluation.

### 7. Additional Practice

`implementing_dummy_gpt_model.ipynb` provides a small GPT-style model implementation for architecture practice.

`Project_1.ipynb` and the SMS spam dataset provide a separate applied machine learning exercise using the included `train.csv`, `validation.csv`, `test.csv`, and `sms_spam_collection/` data.

## Current Scope

- The notebooks are educational and should not be interpreted as a production LLM system.
- The focus is on understanding LLM internals and fine-tuning concepts.
- The repo is best used as a study path before moving to larger instruction-tuned models.

## Recommended Next Steps

1. Choose a stronger instruction model.

   Good next candidates are small instruction-tuned models such as Qwen, Phi, Gemma, Mistral, or Llama variants, depending on available hardware.

2. Build a better instruction dataset.

   Create a larger dataset with consistent fields such as `instruction`, `input`, and `response`. Include examples from the actual domain where the model is expected to help.

3. Add proper evaluation.

   Keep a fixed set of test prompts and compare outputs before and after fine-tuning. Track response quality, factual accuracy, formatting, and refusal behavior where relevant.

4. Use parameter-efficient fine-tuning.

   For larger models, use LoRA or QLoRA instead of full fine-tuning. This is more practical on consumer hardware.

5. Consider retrieval-augmented generation.

   If the model needs to answer from specific documents, add retrieval over trusted files instead of expecting it to memorize facts.

## Final Takeaway

This repo is useful as a bottom-up LLM learning project. Its strongest parts are the notebooks that explain tokenization, embeddings, attention, Transformer basics, and instruction fine-tuning.

The best next move is to improve the fine-tuning path with a stronger instruction model and better data.
