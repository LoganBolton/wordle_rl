import json
import random
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import numpy as np
from wordle_env import WordleEnv


class WordleDataset(Dataset):
    """
    Dataset for multi-turn Wordle RL training.
    
    Unlike traditional datasets, this generates dynamic prompts based on
    ongoing game states that persist across training steps.
    """
    
    def __init__(
        self, 
        num_episodes: int = 1000,
        max_attempts: int = 6,
        tokenizer: PreTrainedTokenizer = None,
        max_prompt_length: int = 2048,
        seed: int = 42
    ):
        self.num_episodes = num_episodes
        self.max_attempts = max_attempts
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        
        # Initialize game environments for each episode
        self.seed = seed
        random.seed(seed)
        self.environments = {}
        self.episode_states = {}
        
        # Initialize all episodes
        self._initialize_episodes()
        
    def _initialize_episodes(self):
        """Initialize all episodes with fresh Wordle games"""
        for i in range(self.num_episodes):
            # Each episode gets a random word (or None for random selection)
            # Use different seeds for each episode to ensure different words
            episode_seed = self.seed + i if hasattr(self, 'seed') else None
            self.environments[i] = WordleEnv(word="tests", max_attempts=self.max_attempts)
            # Set individual seed for each episode's word selection
            if episode_seed is not None:
                import numpy as np
                np.random.seed(episode_seed)
            self.episode_states[i] = {
                'episode_id': i,
                'turn': 0,
                'completed': False,
                'target_word': self.environments[i].game.word,
                'guess_history': [],
                'result_history': []
            }
    
    def __len__(self):
        return self.num_episodes
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get current game state as a prompt for the model.
        
        Returns:
            dict: Contains input_ids, attention_mask, and episode metadata
        """
        env = self.environments[idx]
        state = self.episode_states[idx]
        
        # Generate prompt based on current game state
        prompt = self._generate_game_prompt(env, state)
        
        # Tokenize prompt
        model_inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(0),
            'attention_mask': model_inputs['attention_mask'].squeeze(0),
            'episode_id': idx,
            'turn': state['turn'],
            'completed': state['completed'],
            'target_word': state['target_word'],
            'guess_history': state['guess_history'].copy(),
            'result_history': state['result_history'].copy(),
            'ground_truth': state['target_word']  # For compatibility with VERL
        }
    
    def _generate_game_prompt(self, env: WordleEnv, state: Dict) -> str:
        """Generate a prompt based on current game state"""
        
        if state['turn'] == 0:
            # First turn - introduce the game
            prompt = """You are playing Wordle! Your goal is to guess a 5-letter word.

Rules:
- You have 6 attempts maximum
- After each guess, you'll get feedback:
  - Letters in correct position will be marked as CORRECT
  - Letters in the word but wrong position will be marked as WRONG POSITION  
  - Letters not in the word will be marked as NOT IN WORD

Let's start! What's your first guess?

Your guess:"""
        else:
            # Subsequent turns - show history and ask for next guess
            prompt = "You are playing Wordle! Here's your progress so far:\n\n"
            
            # Add guess history
            for i, (guess, result) in enumerate(zip(state['guess_history'], state['result_history'])):
                prompt += f"Guess {i+1}: {guess.upper()}\n"
                if result:  # Valid guess
                    feedback = []
                    for letter, score in result:
                        if score == 2:
                            feedback.append(f"{letter.upper()}: CORRECT")
                        elif score == 1:
                            feedback.append(f"{letter.upper()}: WRONG POSITION")
                        else:
                            feedback.append(f"{letter.upper()}: NOT IN WORD")
                    prompt += f"Feedback: {' | '.join(feedback)}\n"
                else:
                    prompt += "Invalid guess!\n"
                prompt += "\n"
            
            # Add current status
            remaining = self.max_attempts - state['turn']
            prompt += f"Attempts remaining: {remaining}\n"
            prompt += f"What's your next guess?\n\n"
            prompt += "Your guess:"
            
        return prompt
    
    def update_episode(self, episode_id: int, guess: str) -> Dict[str, Any]:
        """
        Update an episode with a new guess and return the result.
        
        Args:
            episode_id: ID of the episode to update
            guess: Player's guess
            
        Returns:
            dict: Contains reward, done, and updated state info
        """
        env = self.environments[episode_id]
        state = self.episode_states[episode_id]
        
        if state['completed']:
            # Episode already completed, return neutral result
            return {
                'reward': 0.0,
                'done': True,
                'solved': env.game.solved,
                'failed': env.game.failed,
                'info': 'Episode already completed'
            }
        
        # Step the environment
        observation, reward, done, info = env.step(guess)
        
        # Update episode state
        state['turn'] += 1
        state['guess_history'].append(guess)
        state['result_history'].append(env.last_result)
        
        if done:
            state['completed'] = True
            
        return {
            'reward': reward,
            'done': done,
            'solved': info['solved'],
            'failed': info['failed'],
            'observation': observation,
            'info': info
        }
    
    def reset_episode(self, episode_id: int, target_word: Optional[str] = None):
        """Reset a specific episode to start a new game"""
        self.environments[episode_id] = WordleEnv(word=target_word, max_attempts=self.max_attempts)
        self.episode_states[episode_id] = {
            'episode_id': episode_id,
            'turn': 0,
            'completed': False,
            'target_word': self.environments[episode_id].game.word,
            'guess_history': [],
            'result_history': []
        }
    
    def reset_completed_episodes(self):
        """Reset all completed episodes to start new games"""
        for episode_id in range(self.num_episodes):
            if self.episode_states[episode_id]['completed']:
                self.reset_episode(episode_id)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics about current episodes"""
        completed = sum(1 for state in self.episode_states.values() if state['completed'])
        solved = sum(1 for i in range(self.num_episodes) 
                    if self.episode_states[i]['completed'] and self.environments[i].game.solved)
        failed = sum(1 for i in range(self.num_episodes) 
                    if self.episode_states[i]['completed'] and self.environments[i].game.failed)
        
        completed_turns = [state['turn'] for state in self.episode_states.values() if state['completed']]
        avg_turns = np.mean(completed_turns) if completed_turns else 0.0
        
        return {
            'total_episodes': self.num_episodes,
            'completed_episodes': completed,
            'solved_episodes': solved,
            'failed_episodes': failed,
            'success_rate': solved / max(completed, 1),
            'average_turns': float(avg_turns)
        }
    
    def get_active_episodes(self) -> List[int]:
        """Get list of episode IDs that are still active"""
        return [i for i in range(self.num_episodes) if not self.episode_states[i]['completed']]
    
    def get_completed_episodes(self) -> List[int]:
        """Get list of episode IDs that are completed"""
        return [i for i in range(self.num_episodes) if self.episode_states[i]['completed']]


def collate_wordle_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for Wordle batches.
    
    Handles variable-length sequences and episode metadata.
    """
    # Separate tensor and non-tensor data
    tensor_data = {}
    non_tensor_data = {}
    
    # Process each key
    for key in batch[0].keys():
        if key in ['input_ids', 'attention_mask']:
            # Stack tensors with padding
            tensors = [item[key] for item in batch]
            max_len = max(t.size(0) for t in tensors)
            
            padded_tensors = []
            for t in tensors:
                if t.size(0) < max_len:
                    pad_size = max_len - t.size(0)
                    if key == 'input_ids':
                        # Pad input_ids with pad_token_id (usually 0)
                        padded = torch.cat([t, torch.zeros(pad_size, dtype=t.dtype)])
                    else:
                        # Pad attention_mask with 0s
                        padded = torch.cat([t, torch.zeros(pad_size, dtype=t.dtype)])
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(t)
            
            tensor_data[key] = torch.stack(padded_tensors)
        else:
            # Non-tensor data
            non_tensor_data[key] = np.array([item[key] for item in batch], dtype=object)
    
    return {**tensor_data, **non_tensor_data}


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = WordleDataset(
        num_episodes=10,
        tokenizer=tokenizer,
        seed=42
    )
    
    # Test getting items
    print("=== Testing Dataset ===")
    for i in range(3):
        item = dataset[i]
        print(f"\nEpisode {i}:")
        print(f"Target word: {item['target_word']}")
        print(f"Turn: {item['turn']}")
        print(f"Prompt preview: {tokenizer.decode(item['input_ids'][:50])}...")
    
    # Test episode updates
    print("\n=== Testing Episode Updates ===")
    result = dataset.update_episode(0, "CRANE")
    print(f"Guess 'CRANE' result: {result}")
    
    # Get updated prompt
    updated_item = dataset[0]
    print(f"Updated prompt preview: {tokenizer.decode(updated_item['input_ids'][:100])}...")
    
    # Test stats
    print("\n=== Dataset Stats ===")
    print(dataset.get_episode_stats()) 