def wordle_reward(
    data_source: str,
    solution_str: str | None,
    ground_truth: str | None,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """
    Reward function that uses the rewards from WordleInteraction.
    This function will be called after all interactions are completed,
    so we return 0 here to let the interaction-based rewards drive learning.
    
    The actual reward shaping happens in WordleInteraction:
    - +10 for solving the puzzle
    - -5 for failing (running out of attempts)
    - -2 for invalid guesses
    - +1 per correct letter in correct position
    - +0.5 per correct letter in wrong position
    """
    return 0.0