import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.utils.wordle_env import WordleEnv

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def extract_boxed_content(text: str) -> str:
    """Extract the final \boxed{} content from the text.
    
    Args:
        text: Input text that may contain \boxed{content}
        
    Returns:
        Content inside the last \boxed{} tag found, or empty string if no boxed content found
    """
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    if matches:
        final_content = matches[-1].strip()  # Get the last match
        # logger.info(f"Found {len(matches)} boxed content(s), returning final: '{final_content}'")
        return final_content
    logger.warning(f"No boxed content found in text: {repr(text[:100])}...")
    return ""  # Return empty string instead of the entire text


class WordleInteraction(BaseInteraction):
    """Multi‑turn interaction wrapper for training an agent on Wordle.

    Each interaction corresponds to one Wordle episode. The agent issues a
    five‑letter guess as the *user* message; the wrapper returns a textual
    description of the game state as the *assistant* message, plus a shaped
    reward. The episode terminates when the word is solved or the maximum
    number of attempts is reached.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._max_instances = config.get("max_instances", 500)  # Limit concurrent instances

    # ---------------------------------------------------------------------
    # BaseInteraction API
    # ---------------------------------------------------------------------

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        target_word: Optional[str] = None,
        **_: Any,
    ) -> str:
        """Create a fresh Wordle environment and return its *instance_id*."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Cleanup old instances if we're at the limit
        if len(self._instance_dict) >= self._max_instances:
            # Remove oldest half of instances (FIFO cleanup)
            old_keys = list(self._instance_dict.keys())[:-self._max_instances//2]
            for key in old_keys:
                self._instance_dict.pop(key, None)
            logger.info(f"Cleaned up {len(old_keys)} old game instances. Current count: {len(self._instance_dict)}")
        
        # print(f"DEBUG: Starting interaction with target word: {target_word}")
        env = WordleEnv(word=target_word)
        self._instance_dict[instance_id] = {"env": env, "reward": 0.0, "total_reward": 0.0, "all_guesses": set()}
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **_: Any,
    ) -> tuple[bool, str, float, dict]:
        """Advance the game using the latest *user* guess.

        Returns:
            should_terminate_sequence: bool
            response: str                 # assistant reply shown to agent
            reward: float                 # scalar RL reward
            info: dict                    # diagnostics for logging
        """
        # ------------------------------------------------------------------
        # 1. Extract the latest user guess and validate format
        # ------------------------------------------------------------------
        guess = ""
        feedback_message = ""
        raw_guess = ""
        
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                raw_guess = extract_boxed_content(content).lower()
                # Remove dashes to handle tokenization-friendly format (e.g., "s-t-a-r-s" -> "stars")
                clean_guess = raw_guess.replace("-", "").replace(" ", "")
                
                # Check for repeat guess first (before validation)
                if raw_guess in self._instance_dict[instance_id]["all_guesses"]:
                    feedback_message = f"You already guessed '{raw_guess}'. Think about a different word you should guess."
                    # Update total reward for repeat guess penalty
                    self._instance_dict[instance_id]["total_reward"] += -3.0
                    # Create info dict for repeat guess
                    env = self._instance_dict[instance_id]["env"]
                    info = {
                        "error": "repeat_guess",
                        "raw_guess": raw_guess,
                        "attempts": getattr(env, 'attempts', 0),
                        "target_word": getattr(env.game, 'word', None),
                        "total_reward": self._instance_dict[instance_id]["total_reward"],
                        "all_guesses": list(self._instance_dict[instance_id]["all_guesses"])
                    }
                    return False, feedback_message, -3.0, info
                
                # Add to all guesses set
                self._instance_dict[instance_id]["all_guesses"].add(raw_guess)
                
                # Check if it's alphabetic first
                if not clean_guess.isalpha():
                    feedback_message = f"Your guess '{raw_guess}' contains non-letter characters. Please use only letters or dashes. Guess a different word."
                    break
                elif len(clean_guess) < 5:
                    feedback_message = f"Your guess '{raw_guess}' has only {len(clean_guess)} letters. Wordle words must be exactly 5 letters. Guess a different word."
                    break
                elif len(clean_guess) > 5:
                    feedback_message = f"Your guess '{raw_guess}' has {len(clean_guess)} letters. Wordle words must be exactly 5 letters. Guess a different word."
                    break
                else:
                    # Valid 5-letter word
                    guess = clean_guess
                    break
            pass
        
        # If no valid guess found, provide specific feedback
        if not guess:
            if not feedback_message:
                feedback_message = "Please provide a 5-letter word guess using the format: L-E-T-T-E-R"
            # Update total reward for invalid format penalty
            self._instance_dict[instance_id]["total_reward"] += -1.0
            # Create info dict for invalid format
            env = self._instance_dict[instance_id]["env"]
            info = {
                "error": "invalid_format",
                "raw_guess": raw_guess,
                "attempts": getattr(env, 'attempts', 0),
                "target_word": getattr(env.game, 'word', None),
                "total_reward": self._instance_dict[instance_id]["total_reward"],
                "all_guesses": list(self._instance_dict[instance_id]["all_guesses"])
            }
            return False, feedback_message, -1.0, info

        env: WordleEnv = self._instance_dict[instance_id]["env"]

        # ------------------------------------------------------------------
        # 2. Step the environment and compute reward
        # ------------------------------------------------------------------
        observation, reward, done, info = env.step(guess)
        self._instance_dict[instance_id]["reward"] = reward
        # Accumulate total reward across all turns
        self._instance_dict[instance_id]["total_reward"] += reward

        # ------------------------------------------------------------------
        # 3. Build assistant message (state prompt)
        # ------------------------------------------------------------------
        response = env.get_state_prompt(guess)
        should_terminate_sequence = done

        return should_terminate_sequence, response, reward, info

    async def calculate_score(self, instance_id: str, **_: Any) -> float:
        """Return the accumulated total reward across all turns."""
        return float(self._instance_dict[instance_id]["total_reward"])

    async def finalize_interaction(self, instance_id: str, **_: Any) -> None:
        """Clean up the interaction state."""
        self._instance_dict.pop(instance_id, None)
