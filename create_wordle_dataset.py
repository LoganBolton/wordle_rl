#!/usr/bin/env python3
"""
Create a proper Wordle dataset for VERL training
"""
import pandas as pd
import os

def create_wordle_dataset():
    """Create a minimal Wordle dataset with the required VERL structure"""
    
    # Common 5-letter words for Wordle
    target_words = ["HOUSE", "APPLE"]
    
    # Create some sample Wordle prompts
    data = []
    
    for i in range(10):  # Create 10 sample entries
        entry = {
            "data_source": "wordle",  # This is the key the reward manager looks for
            "prompt": [
                {
                    "role": "system", 
                    "content": "You are an agent playing Wordle. Your job is to guess a real 5-letter word. Your output should ONLY be a 5-letter word and NOTHING ELSE. You should respond with dashes separrating each letter. For example, if your guess is 'HELLO', you should respond with 'H-E-L-L-O'."
                },
                {
                    "role": "user", 
                    "content": "Respond with your new one word guess."
                }
            ],
            "ability": "game",
            "reward_model": {
                "style": "rule",
                "ground_truth": target_words[i%len(target_words)]
            },
            "extra_info": {
                "index": i,
                "split": "train",
                "game_id": f"wordle_{i:04d}",
                "interaction_kwargs": {
                    "name": "wordle",
                    "target_word": target_words[i%len(target_words)]
                }
            }
        }
        data.append(entry)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs("/tmp/wordle_data", exist_ok=True)
    
    # Save as parquet
    df.to_parquet("/tmp/wordle_data/wordle_dataset.parquet", index=False)
    print(f"Created wordle dataset with {len(data)} entries")
    print("Dataset structure:")
    print(df.columns.tolist())
    print("\nFirst entry:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    create_wordle_dataset() 