#!/usr/bin/env python3
"""
Create a proper Wordle dataset for VERL training
"""
import pandas as pd
import os

def create_wordle_dataset():
    """Create a minimal Wordle dataset with the required VERL structure"""
    
    # Common 5-letter words for Wordle
    target_words = [
        "house", "apple", "class", "train", "brain", "chain", "chair", "claim",
        "charm", "chart", "chase", "cheap", "check", "chess", "chest", "chief",
        "child", "china", "clock", "close", "cloud", "coach", "coast", "couch",
        "count", "court", "crash", "crazy", "cream", "crime", "cross", "crowd",
        "crown", "dance", "death", "doubt", "dream", "dress", "drink", "drive",
        "earth", "empty", "enemy", "enjoy", "enter", "equal", "error", "event",
        "every", "exact", "exist", "extra", "faith", "false", "field", "fight",
        "final", "first", "flash", "floor", "focus", "force", "frame", "fresh",
        "front", "fruit", "funny", "ghost", "giant", "glass", "grace", "grand",
        "grant", "grass", "great", "green", "gross", "group", "guess", "guest",
        "guide", "happy", "heart", "heavy", "horse", "hotel", "human", "hurry",
        "image", "index", "inner", "input", "issue", "japan", "joint", "judge",
        "knife", "known", "large", "laser", "later", "laugh", "layer", "learn",
        "least", "leave", "legal", "level", "light", "limit", "local", "loose",
        "lower", "lucky", "lunch", "magic", "major", "maker", "march", "match",
        "maybe", "means", "media", "metal", "might", "minor", "minus", "mixed",
        "model", "money", "month", "moral", "motor", "mount", "mouse", "mouth",
        "movie", "music", "needs", "never", "night", "noise", "north", "novel",
        "nurse", "occur", "ocean", "offer", "often", "order", "other", "ought",
        "outer", "owner", "paint", "panel", "paper", "party", "peace", "peter",
        "phase", "phone", "photo", "piano", "piece", "pilot", "pitch", "place",
        "plain", "plane", "plant", "plate", "point", "pound", "power", "press",
        "price", "pride", "prime", "print", "prior", "prize", "proof", "proud",
        "prove", "queen", "quick", "quiet", "quite", "radio", "raise", "range",
        "rapid", "ratio", "reach", "ready", "realm", "rebel", "refer", "relax",
        "reply", "right", "rival", "river", "robin", "robot", "roger", "roman",
        "rough", "round", "route", "royal", "rural", "scale", "scene", "scope",
        "score", "sense", "serve", "setup", "seven", "shade", "shake", "shall",
        "shame", "shape", "share", "sharp", "sheet", "shelf", "shell", "shine",
        "shirt", "shock", "shoot", "short", "shown", "sides", "sight", "silly",
        "since", "sixth", "sixty", "sized", "skill", "sleep", "slide", "small",
        "smart", "smile", "smoke", "snake", "snow", "solid", "solve", "sorry",
        "sound", "south", "space", "spare", "speak", "speed", "spend", "spent",
        "split", "spoke", "sport", "staff", "stage", "stake", "stand", "start",
        "state", "steam", "steel", "stick", "still", "stock", "stone", "stood",
        "store", "storm", "story", "strip", "stuck", "study", "stuff", "style",
        "sugar", "suite", "super", "sweet", "table", "taken", "taste", "taxes",
        "teach", "terms", "thank", "theft", "their", "theme", "there", "these",
        "thick", "thing", "think", "third", "those", "three", "threw", "throw",
        "thumb", "tight", "timer", "tired", "title", "today", "topic", "total",
        "touch", "tough", "tower", "track", "trade", "train", "treat", "trend",
        "trial", "tribe", "trick", "tried", "tries", "truly", "trunk", "trust",
        "truth", "twice", "under", "undue", "union", "unity", "until", "upper",
        "urban", "usage", "usual", "valid", "value", "video", "virus", "visit",
        "vital", "vocal", "voice", "waste", "watch", "water", "wheel", "where",
        "which", "while", "white", "whole", "whose", "wider", "woman", "women",
        "world", "worry", "worse", "worst", "worth", "would", "write", "wrong",
        "wrote", "young", "youth", "zones"
    ]
    
    # Create some sample Wordle prompts
    data = []
    
    for i in range(200):  # Create N sample entries
        entry = {
            "data_source": "wordle",  # This is the key the reward manager looks for
            "prompt": [
                {
                    "role": "system", 
                    "content": f"You are an agent playing Wordle. Your job is to guess a real 5-letter word. At the start of every turn, think step by step of what word you should guess. Format your final one word answer in \\boxed{{}}"
                },
                {
                    "role": "user", 
                    "content": "Respond with your initial guess to start the game."
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
    df.to_parquet("/tmp/wordle_data/train_wordle_dataset.parquet", index=False)
    print(f"Created wordle dataset with {len(data)} entries")
    print("Dataset structure:")
    print(df.columns.tolist())
    print("\nFirst entry:")
    print(df.iloc[0].to_dict())


    # --- Create a small validation set of size 4 ---
    val_data = []
    for i in range(4):
        entry = data[i].copy()
        entry["extra_info"] = entry["extra_info"].copy()
        entry["extra_info"]["split"] = "val"
        entry["extra_info"]["game_id"] = f"wordle_val_{i:04d}"
        entry["extra_info"]["index"] = i
        val_data.append(entry)
    val_df = pd.DataFrame(val_data)
    val_df.to_parquet("/tmp/wordle_data/val_wordle_dataset.parquet", index=False)
    print(f"\nCreated validation wordle dataset with {len(val_data)} entries")
    print("Validation dataset structure:")
    print(val_df.columns.tolist())
    print("\nFirst val entry:")
    print(val_df.iloc[0].to_dict())

if __name__ == "__main__":
    create_wordle_dataset() 