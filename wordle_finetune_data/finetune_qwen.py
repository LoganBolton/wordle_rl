#!/usr/bin/env python3
"""
Qwen3-1.7B Finetuning Script for Wordle Dataset
Full parameter finetuning (not LoRA) with wandb monitoring
"""
import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from dataclasses import dataclass
import wandb
from typing import Dict, List


@dataclass
class DataCollatorForCausalLM:
    """Custom data collator for causal language modeling"""
    tokenizer: AutoTokenizer
    max_length: int = 2048
    
    def __call__(self, features):
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences to the same length
        max_len = min(max(len(seq) for seq in input_ids), self.max_length)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, lbls in zip(input_ids, labels):
            # Truncate if necessary
            if len(ids) > max_len:
                ids = ids[:max_len]
                lbls = lbls[:max_len]
            
            # Pad sequences
            pad_length = max_len - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * pad_length
            padded_lbls = lbls + [-100] * pad_length  # -100 is ignored in loss calculation
            attention_mask = [1] * len(ids) + [0] * pad_length
            
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbls)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }


def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load JSONL data from file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_chat_messages(messages: List[Dict]) -> str:
    """Convert messages to Qwen3 chat format"""
    formatted_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return formatted_text


def preprocess_function(examples, tokenizer, max_length=2048):
    """Preprocess examples for training"""
    texts = []
    for messages in examples["messages"]:
        formatted_text = format_chat_messages(messages)
        texts.append(formatted_text)
    
    # Tokenize the texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = [input_ids[:] for input_ids in tokenized["input_ids"]]
    
    return tokenized


def main():
    # Configuration
    model_name = "Qwen/Qwen3-1.7B"
    data_file = "wordle_finetune.jsonl"
    output_dir = "./qwen_wordle_finetuned"
    
    # Initialize wandb
    wandb.init(
        project="qwen-wordle-finetune",
        name="qwen-1.7b-wordle-full-finetune",
        config={
            "model": model_name,
            "dataset": "wordle_synthetic",
            "training_type": "full_finetune"
        }
    )
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Load and prepare dataset
    print("Loading dataset...")
    raw_data = load_jsonl_data(data_file)
    
    # Extract messages for dataset
    dataset_dict = {"messages": [item["messages"] for item in raw_data]}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset (90% train, 10% eval)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        max_length=2048
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=1,
        eval_steps=20,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_drop_last=True,
        bf16=True,
        gradient_checkpointing=True,
        report_to=["wandb"],
        logging_first_step=True,
        run_name="qwen-wordle-finetune",
        save_total_limit=3,
        prediction_loss_only=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Finish wandb run
    wandb.finish()
    
    print(f"Training completed! Model saved to {output_dir}")


if __name__ == "__main__":
    main()