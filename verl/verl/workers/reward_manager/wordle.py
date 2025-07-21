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
                # Sum over all user-turn rewards (can switch to last reward if
                # desired).
                user_turn_rewards = reward_dict.get("user_turn_rewards", [])
                if len(user_turn_rewards) > 0:
                    sample_rewards = float(sum(user_turn_rewards))
                
                # Extract metrics from user turns for extra info
                user_turn_metrics = reward_dict.get("user_turn_metrics", [])
                for turn_idx, metrics in enumerate(user_turn_metrics):
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            # Convert complex objects to strings for numpy compatibility
                            if isinstance(value, list):
                                value = str(value)  # Convert lists to strings
                            elif isinstance(value, dict):
                                value = str(value)  # Convert dicts to strings
                            elif value is None:
                                value = ""  # Convert None to empty string
                            
                            if key not in reward_extra_info:
                                reward_extra_info[key] = [""] * i  # Pad previous samples with empty strings
                            # Ensure we have enough space for current sample
                            while len(reward_extra_info[key]) < i:
                                reward_extra_info[key].append("")
                            if len(reward_extra_info[key]) == i:
                                reward_extra_info[key].append(str(value))
                            else:
                                reward_extra_info[key][i] = str(value)
            
            # Place the scalar reward on the last valid response token.
            last_token_idx = int(response_mask[i].sum().item()) - 1
            if last_token_idx >= 0:
                reward_tensor[i, last_token_idx] = sample_rewards
            # Store extra diagnostics if requested.
            reward_extra_info["env_return"].append(sample_rewards)
            
            # Ensure all reward_extra_info lists have the same length
            for key in reward_extra_info:
                while len(reward_extra_info[key]) <= i:
                    reward_extra_info[key].append("")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor 