from typing import Any, Dict, Tuple, Union, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.utils.wordle_env import WordleEnv

class WordleTool(BaseTool):
    """Verl tool that lets an LLM play Wordle in a multi-turn loop."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, WordleEnv] = {}

    # ───────────────────────── create ─────────────────────────
    async def create(
        self,
        instance_id: Optional[str] = None,
        word: Optional[str] = None,
        **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        env = WordleEnv(word=word, max_attempts=self.config.get("max_attempts", 6))
        self._instance_dict[instance_id] = env
        return instance_id

    # ───────────────────────── execute ────────────────────────
    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Tuple[Union[str, Dict[str, Any]], float, dict]:
        guess: str = parameters.get("guess", "").lower()
        env = self._instance_dict[instance_id]

        obs, reward, done, info = env.step(guess)

        # Verl expects the first element to be *assistant-visible* text or a multimodal dict.
        tool_response = env.get_state_prompt(guess)

        # You can add done flag to info so the collector knows when to stop even
        # if you set a very large max_steps.
        info["done"] = done
        return tool_response, reward, info

    # ──────────────────────── calc_reward ─────────────────────
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Optional—Verl will call this at episode end if configured."""
        env = self._instance_dict[instance_id]
        # Final reward = +10 if solved, else −5 (your env already sets this)
        return env.get_reward()

    # ───────────────────────── release ────────────────────────
    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)