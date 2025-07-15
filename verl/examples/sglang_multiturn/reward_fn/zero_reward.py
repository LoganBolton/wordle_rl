def zero_reward(
    data_source: str,
    solution_str: str | None,
    ground_truth: str | None,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """
    Return 0 for every sample so that PPO learns only from the
    online reward produced inside WordleInteraction.
    """
    return 0.0