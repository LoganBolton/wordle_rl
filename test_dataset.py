#!/usr/bin/env python3
"""
Simple test script for the Wordle dataset with hardcoded word
"""
import sys
sys.path.append('.')

from wordle_dataset import WordleDataset

def test_wordle_dataset():
    """Test the Wordle dataset functionality"""
    print("=== Testing Wordle Dataset ===")
    
    # Create a small dataset for testing
    dataset = WordleDataset(num_episodes=3, max_attempts=6)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Target word: {dataset.episode_states[0]['target_word']}")
    
    # Test getting initial prompts
    print("\n=== Initial Prompts ===")
    for i in range(3):
        item = dataset[i]
        print(f"Episode {i}: Turn {item['turn']}, Target: {item['target_word']}")
    
    # Test episode updates
    print("\n=== Testing Episode Updates ===")
    
    # Test with a bad guess
    result = dataset.update_episode(0, "TESTS")
    print(f"Guess 'TESTS' on CRANE: reward={result['reward']}, done={result['done']}")
    
    # Test with the correct guess
    result = dataset.update_episode(1, "CRANE")
    print(f"Guess 'CRANE' on CRANE: reward={result['reward']}, done={result['done']}, solved={result['solved']}")
    
    # Test getting updated prompts
    print("\n=== Updated Prompts ===")
    updated_item = dataset[0]
    print(f"Episode 0 after guess: Turn {updated_item['turn']}, History: {updated_item['guess_history']}")
    
    completed_item = dataset[1]
    print(f"Episode 1 after solving: Turn {completed_item['turn']}, Completed: {completed_item['completed']}")
    
    # Test stats
    print("\n=== Dataset Stats ===")
    stats = dataset.get_episode_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test reset
    print("\n=== Testing Reset ===")
    dataset.reset_completed_episodes()
    print(f"Active episodes after reset: {dataset.get_active_episodes()}")
    print(f"Completed episodes after reset: {dataset.get_completed_episodes()}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_wordle_dataset() 