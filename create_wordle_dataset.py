#!/usr/bin/env python3
"""
Create a proper Wordle dataset for VERL training
"""
import pandas as pd
import os

def create_wordle_dataset():
    """Create a minimal Wordle dataset with the required VERL structure"""
    
    # Create some sample Wordle prompts
    data = []
    
    for i in range(10):  # Create 10 sample entries
        entry = {
            "data_source": "wordle",  # This is the key the reward manager looks for
            "prompt": [
                {
                    "role": "system", 
                    "content": "You are playing Wordle. Guess a 5-letter word. You'll receive feedback for each guess."
                },
                {
                    "role": "user", 
                    "content": "Let's start a new Wordle game! Make your first guess."
                }
            ],
            "ability": "game",
            "reward_model": {
                "style": "rule",
                "ground_truth": "WORLD"  # Example target word - this should be replaced with actual game logic
            },
            "extra_info": {
                "index": i,
                "split": "train",
                "game_id": f"wordle_{i:04d}"
            }
        }
        data.append(entry)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save as parquet
    df.to_parquet("data/wordle_dataset.parquet", index=False)
    print(f"Created wordle dataset with {len(data)} entries")
    print("Dataset structure:")
    print(df.columns.tolist())
    print("\nFirst entry:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    create_wordle_dataset() 