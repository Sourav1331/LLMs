# Final Report: LLMs From Scratch to Deployment

## Project Overview

This project is a hands-on journey through the core ideas behind Large Language Models. It starts with tokenization from scratch, builds up through embeddings, attention, Transformer blocks, and instruction fine-tuning, then finishes with a deployable API around a fine-tuned model checkpoint.

The goal is not to compete with production-scale models such as GPT, Llama, Qwen, Mistral, or Gemma. The goal is to understand the full pipeline deeply enough to build, fine-tune, save, reload, and serve a small LLM-style system.

## Repository Structure

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
  app/
  fine_tuned_distilgpt2_instruction_demo/
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

The project begins by converting raw text into tokens. Tokenization is the first step in any LLM pipeline because neural networks cannot directly process raw text. The early notebooks show how text can be split, indexed, encoded, and decoded.

### 2. Byte Pair Encoding

Byte Pair Encoding improves simple tokenization by learning frequent character or subword merges. This makes the tokenizer more flexible because it can handle common words efficiently while still representing rare words through smaller pieces.

### 3. Input-Target Pairs

Language models are trained to predict the next token. The input-target pair notebooks show how a text sequence can be converted into examples where the input is a context window and the target is the shifted next-token sequence.

### 4. Token Embeddings

Tokens are integer IDs, but models need dense vectors. Embeddings convert token IDs into learnable vector representations. These vectors become the model's internal representation of language.

### 5. Positional Embeddings

Transformers process tokens in parallel, so they need extra information about token order. Positional embeddings add sequence position information so the model can distinguish between the same words appearing in different places.

### 6. Data Preprocessing

The preprocessing notebook connects the earlier steps into a repeatable training pipeline. It prepares datasets, tokenizes text, creates batches, and organizes data into a form suitable for training.

### 7. Attention Mechanism

Attention is the key mechanism behind modern LLMs. It allows each token to look at other tokens in the sequence and decide which context matters most. This project covers scaled dot-product attention and the move toward multi-head attention.

### 8. Transformer Basics

The Transformer notebooks combine embeddings, attention, feed-forward layers, residual connections, and normalization into the architecture used by GPT-style models.

### 9. Instruction Fine-Tuning

Instruction fine-tuning teaches a model to respond to user-style prompts instead of only predicting generic next tokens. This is the practical bridge between a language model and an assistant-like system.

### 10. Fine-Tuning DataLoader and Training

The fine-tuning training notebooks cover dataset formatting, batching, masking labels, training loops, validation loss, and checkpoint saving.

### 11. Evaluation and Saved Checkpoint

The final training notebook saves a fine-tuned DistilGPT2-style checkpoint into:

```text
fine_tuned_distilgpt2_instruction_demo/
```

This folder contains the model weights, tokenizer files, configuration, generation settings, and training history.

## Training Result

The saved training history shows improvement across three epochs:

| Epoch | Train Loss | Validation Loss | Validation Perplexity |
|---|---:|---:|---:|
| 1 | 4.7877 | 3.8142 | 45.3395 |
| 2 | 4.4029 | 3.6541 | 38.6316 |
| 3 | 4.0682 | 3.5354 | 34.3079 |

The validation loss and perplexity both decreased, which indicates that the fine-tuning process improved the model on the validation set.

## Deployment Layer

The `app/` folder adds a simple deployment layer using FastAPI.

```text
app/
  api.py
  requirements.txt
  README.md
```

The API loads the saved checkpoint from:

```text
fine_tuned_distilgpt2_instruction_demo/
```

It exposes:

- `GET /` for basic service information.
- `GET /health` for a model-loading health check.
- `POST /generate` for text generation.

This turns the project from a notebook-only learning folder into a small deployable LLM assistant.

## From-Scratch Approach vs Practical Production Approach

Building an LLM completely from scratch is useful for learning because it explains every internal part of the system. However, training a high-quality model from scratch requires massive datasets, expensive GPUs, long training time, and careful evaluation.

The practical modern approach is:

1. Start with a pretrained model.
2. Fine-tune it on a specific dataset.
3. Optionally add retrieval-augmented generation for private knowledge.
4. Evaluate the outputs.
5. Deploy through an API.

This project follows that practical path at the end by fine-tuning an existing DistilGPT2 checkpoint and wrapping it with an API.

## Current Limitations

- The fine-tuned model is small and should be treated as a demo model.
- DistilGPT2 is not instruction-following by default, so output quality may be limited.
- The API does not include authentication, rate limiting, monitoring, or logging.
- The current deployment is local-first and should be hardened before public use.
- More data and stronger models would be needed for a production assistant.

## Future Improvements

- Fine-tune a stronger instruction model such as Qwen, Phi, Gemma, Mistral, or Llama.
- Use LoRA or QLoRA for efficient fine-tuning.
- Add retrieval-augmented generation over custom documents.
- Add a frontend chat interface.
- Add Docker deployment.
- Add automated tests for the API.
- Add model evaluation scripts with repeatable benchmark prompts.
- Track experiments with training metrics and model versions.

## Final Takeaway

This folder now represents a complete learning-to-deployment pipeline. It explains LLM internals from the bottom up, demonstrates fine-tuning, saves a model checkpoint, and provides a deployable API for generation.

The strongest next step is to repeat the same deployment flow with a more capable open-source instruction model and a better domain-specific dataset.
