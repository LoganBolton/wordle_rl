from __future__ import annotations

import torch
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register


@register("wordle")
class WordleRewardManager:
    """Reward manager that reads per-episode Wordle scores emitted by the
    multi-turn rollout pipeline (``reward_scores`` in ``non_tensor_batch``).

    Each sample produced by the rollout worker contains a ``reward_scores``
    dict with the following structure::

        {
            "user_turn_rewards": [r_1, r_2, ..., r_T]
        }

    where *r_t* is the scalar reward returned by the Wordle environment after
    turn *t* (positive for progress / success, negative for failure, 0
    otherwise).  We convert these scores into a *token-level* reward tensor
    expected by the PPO trainer: a float tensor with the same shape as the
    ``responses`` field where the **final response token** stores the
    episode-level reward (sum over turns by default).
    """

    def __init__(self, tokenizer=None, num_examine: int = 0, **_):
        # Tokenizer and printing inspection are not strictly necessary for this
        # reward manager, but we preserve the arguments for compatibility with
        # the existing reward-loading utility.
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        
        # Track completion statistics for percent_completed metric
        self.total_games = 0
        self.completed_games = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def __call__(self, data: DataProto, return_dict: bool = False):
        # Pre-allocate zero reward tensor: (bs, response_len)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Response mask to locate the final generated token for each sample.
        if "response_mask" in data.batch:
            response_mask = data.batch["response_mask"]
        else:
            # Fallback: build mask from attention_mask
            responses = data.batch["responses"]
            response_len = responses.size(1)
            attention_mask = data.batch["attention_mask"]
            response_mask = attention_mask[:, -response_len:]

        batch_size = len(data)
        for i in range(batch_size):
            sample_rewards = 0.0
            reward_dict = data.non_tensor_batch["reward_scores"][i]
            if isinstance(reward_dict, dict):
                # Use interaction final score for cumulative rewards
                sample_rewards = float(reward_dict.get("interaction_final_score", 0.0))
                
                # Track game completion for percent_completed metric
                user_turn_metrics = reward_dict.get("user_turn_metrics", [])
                game_completed = False
                
                # Check if any turn shows the game was solved
                for turn_metrics in user_turn_metrics:
                    if isinstance(turn_metrics, dict) and turn_metrics.get("solved", False):
                        game_completed = True
                        break
                
                # Update completion statistics
                self.total_games += 1
                if game_completed:
                    self.completed_games += 1
                
                # Extract metrics from user turns for extra info
                for turn_idx, metrics in enumerate(user_turn_metrics):
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            # Keep numeric values as numeric for charting, convert others to strings
                            if isinstance(value, (int, float, bool)):
                                # Keep numeric values as-is for chart processing
                                processed_value = float(value) if isinstance(value, bool) else value
                            elif isinstance(value, list):
                                processed_value = str(value)  # Convert lists to strings
                            elif isinstance(value, dict):
                                processed_value = str(value)  # Convert dicts to strings
                            elif value is None:
                                processed_value = 0.0  # Convert None to 0 for numeric compatibility
                            else:
                                processed_value = str(value)  # Convert other types to strings
                            
                            if key not in reward_extra_info:
                                # Pad previous samples with appropriate default values
                                default_val = 0.0 if isinstance(processed_value, (int, float)) else ""
                                reward_extra_info[key] = [default_val] * i
                            # Ensure we have enough space for current sample
                            while len(reward_extra_info[key]) < i:
                                default_val = 0.0 if isinstance(processed_value, (int, float)) else ""
                                reward_extra_info[key].append(default_val)
                            if len(reward_extra_info[key]) == i:
                                reward_extra_info[key].append(processed_value)
                            else:
                                reward_extra_info[key][i] = processed_value
            
            # Place the scalar reward on the last valid response token.
            last_token_idx = int(response_mask[i].sum().item()) - 1
            if last_token_idx >= 0:
                reward_tensor[i, last_token_idx] = sample_rewards
            # Store extra diagnostics if requested.
            reward_extra_info["env_return"].append(sample_rewards)
            
            # Ensure all reward_extra_info lists have the same length
            for key in reward_extra_info:
                while len(reward_extra_info[key]) <= i:
                    # Use appropriate default value based on existing data type
                    if len(reward_extra_info[key]) > 0:
                        default_val = 0.0 if isinstance(reward_extra_info[key][0], (int, float)) else ""
                    else:
                        default_val = ""
                    reward_extra_info[key].append(default_val)
        
        # Add percent_completed metric - use cumulative completion rate as a numeric value
        if "percent_completed" not in reward_extra_info:
            # Calculate the current cumulative completion rate
            percent_completed = float((self.completed_games / self.total_games * 100.0) if self.total_games > 0 else 0.0)
            reward_extra_info["percent_completed"] = [percent_completed] * batch_size

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
    
    def get_completion_stats(self):
        """Get current completion statistics"""
        percent_completed = (self.completed_games / self.total_games * 100.0) if self.total_games > 0 else 0.0
        return {
            "total_games": self.total_games,
            "completed_games": self.completed_games,
            "percent_completed": percent_completed
        }
    
    def reset_completion_stats(self):
        """Reset completion statistics"""
        self.total_games = 0
        self.completed_games = 0