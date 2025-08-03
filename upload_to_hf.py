#!/usr/bin/env python3
"""
Upload finetuned Qwen model to Hugging Face Hub
"""
from huggingface_hub import HfApi, create_repo
import os

def upload_model():
    # Configuration
    model_dir = "./checkpoints/verl_wordle/wordle-qwen2.5-0.5b/global_step_2000/actor"
    repo_name = "qwen-1.7b-wordle-grpo"
    
    # Get your username
    username = "loganbolton" 
    repo_id = f"{username}/{repo_name}"
    
    # Initialize API
    api = HfApi()
    
    # Create repository (if it doesn't exist)
    try:
        create_repo(repo_id=repo_id, repo_type="model")
        print(f"Created repository: {repo_id}")
    except Exception as e:
        print(f"Repository may already exist: {e}")
    
    # Upload the model
    print(f"Uploading model to {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"Model uploaded successfully!")
    print(f"View your model at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    upload_model()