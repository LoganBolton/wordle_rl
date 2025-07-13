#!/usr/bin/env python3
"""
Simple test script to verify the Qwen2.5-VL model can be loaded correctly
in the Docker environment before running the full training.
"""

import torch
import os
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

def test_model_loading():
    print("Testing Qwen2.5-VL model loading...")
    
    # Set environment variables for stability
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
    
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    try:
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}, Memory: {gpu_memory:.2f} GB")
        else:
            print("CUDA is not available")
            return False
        
        # Load tokenizer and processor
        print("Loading tokenizer and processor...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
        
        # Load model with conservative memory settings
        print("Loading model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
            attn_implementation="flash_attention_2"
        )
        
        print("Model loaded successfully!")
        
        # Test basic functionality
        print("Testing basic tokenization...")
        test_text = "Hello, world!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"Tokenized text: {tokens}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        print("✅ Model loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1) 