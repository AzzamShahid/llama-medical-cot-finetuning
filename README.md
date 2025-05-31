# Parameter-Efficient Fine-Tuning of LLaMA 3.2 (3B) on Medical Chain-of-Thought Dataset

## Overview

This repository contains the implementation of parameter-efficient supervised fine-tuning of the LLaMA 3.2 (3B) model using a medical Chain-of-Thought (CoT) dataset. The fine-tuning process employs Low-Rank Adaptation (LoRA) techniques through the Unsloth framework to enhance the model's ability to generate step-by-step medical reasoning and structured responses.

## Objective

The primary objective is to fine-tune the LLaMA 3.2 (3B) model on medical Chain-of-Thought data while applying parameter-efficient fine-tuning techniques. The model is trained to generate structured medical responses with explicit reasoning steps, improving its capability for medical question-answering tasks.

## Dataset

The project utilizes the FreedomIntelligence/medical-o1-reasoning-SFT dataset from Hugging Face, which contains medical questions with corresponding Chain-of-Thought reasoning and responses. The dataset is formatted with:

- **Think tags**: Step-by-step reasoning enclosed in `<think>...</think>`
- **Response tags**: Final answers enclosed in `<response>...</response>`

### Dataset Split
- Training data: All samples except the first 100
- Validation data: First 100 samples

## Model Architecture

- **Base Model**: LLaMA 3.2 (3B) Instruct (unsloth/llama-3.2-3b-Instruct)
- **Quantization**: 4-bit quantization for memory efficiency
- **Fine-tuning Method**: Low-Rank Adaptation (LoRA)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Technical Specifications

### LoRA Configuration
- Rank (r): 16
- Alpha: 16
- Dropout: 0
- Bias: none

### Training Parameters
- Epochs: 1
- Batch size per device: 2
- Gradient accumulation steps: 8
- Learning rate: 5e-4
- Weight decay: 0.01
- Max sequence length: 2048
- Warmup steps: 25

## Requirements

### Environment Setup
- Python 3.8+
- CUDA-compatible GPU
- Kaggle Notebooks environment

### Dependencies
```bash
pip install unsloth
pip install bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton
pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
pip install transformers==4.51.3
pip install rouge_score evaluate
```

### API Keys Required
- Hugging Face Token (HF_TOKEN)
- Weights & Biases API Key (WANDB_API_KEY)

## Installation and Usage

### 1. Environment Setup
```python
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
```

### 2. Dataset Loading and Formatting
```python
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
# Dataset formatting and preprocessing as implemented in the notebook
```

### 3. Model Loading
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-3b-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

### 4. LoRA Configuration
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
```

### 5. Training Process
The training process includes:
- Supervised fine-tuning using SFTTrainer
- Weights & Biases integration for monitoring
- Gradient checkpointing for memory optimization
- Regular evaluation and model saving

### 6. Inference
```python
def generate_medical_answer(question, model, tokenizer):
    prompt = f"""Below is a medical question. Think step by step to solve it.
Question: {question}
"""
    # Generation logic as implemented in the notebook
```

## Model Performance

The model performance is evaluated using:
- Training loss monitoring
- Validation loss tracking
- ROUGE-L score comparison (before and after fine-tuning)

All metrics are logged to Weights & Biases for comprehensive monitoring.

## Model Deployment

### Local Saving
The fine-tuned model components are saved locally:
- LoRA adapter weights
- Tokenizer files
- Evaluation results

### Hugging Face Upload
The model is uploaded to Hugging Face Hub at: `AzzamShahid/llama-3b-medical-cot`

### Loading Fine-tuned Model
```python
# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-3b-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load LoRA adapter
model = FastLanguageModel.get_peft_model(model, ...)
model.load_adapter("AzzamShahid/llama-3b-medical-cot")
```

## Monitoring and Logging

The training process is monitored using Weights & Biases, tracking:
- Training and validation loss
- GPU memory consumption
- Training progress metrics
- Model performance evaluation

## Files Structure

The implementation generates the following outputs:
- Fine-tuned model files (LoRA adapter)
- Tokenizer files
- Evaluation results (JSON format)
- Training logs and metrics

## Results

The fine-tuned model demonstrates improved performance in:
- Medical question comprehension
- Step-by-step reasoning generation
- Structured response formatting
- Chain-of-thought medical reasoning

## Technical Notes

- The implementation uses 4-bit quantization for memory efficiency
- Gradient checkpointing is enabled to handle memory constraints
- Parameter-efficient fine-tuning reduces computational requirements
- The model maintains the original LLaMA 3.2 architecture while adapting specific layers

## Validation

The model is tested with sample medical questions covering:
- Acute chest pain differential diagnosis
- Cardiac symptoms evaluation
- Diabetes mellitus classification

Each test demonstrates the model's ability to provide structured, step-by-step medical reasoning.

## License

This project follows the licensing terms of the underlying models and datasets used.
