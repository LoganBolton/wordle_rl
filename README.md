# Wordle project

This project finetunes and uses reinforcement learning to train Qwen3-1.7B to play Wordle. This is a uniquely challenging task as most LLMs are heavily optimized for single turn conversations and NOT conversations that require reasoning over multi turn feedback. This project forks ByteDance's verl repo as at the time of working on this project, verl was essentially the only standard framework that easily supported multiturn RL. 

# Results
| Base Model   | Finetuned?  | GRPO?  | % of Games Succesfully Completed  |
|--------------|-------------|--------|-----------------------------------|
| Qwen3-1.7B   | No ❌       | No ❌  | 0.1                               |
| Qwen3-1.7B   | No ❌       | Yes ✅ | 1.2                               |
| Qwen3-1.7B   | Yes ✅      | No ❌  | 9.3                               |
| Qwen3-1.7B   | Yes ✅      | Yes ✅ | 17.9                              |

# Lessons Learned from this Project

**1. Negative rewards heavily promote reward hacking**

Finding a negative reward is extremely easy for the model. However, finding the major bonus reward for successfully completing a task is significantly harder. Therefore, some easy low-hanging fruit in RL training is that your model will tend to heavily try to avoid those obvious negative rewards. This unfortunately leads to reward-hacking behavior. For example, a frequent problem I had is that the model learned that repeating previously guessed words would result in a negative reward. This caused it to learn to just repeat words over and over to hit the context limit and crash the game in order to avoid the chance of having this negative reward.  

For multi-turn RL, it's actually better to just give a reward of 0 for “bad” behavior and then give a small positive reward for anything else. This encourages the model to explore more heavily instead of just exploiting the fact that it can avoid negative rewards.  


**2. KL Divergence matters a lot!**

Before working on this project, I had read many RL papers that yapped on and on about KL divergence from the base model and different strategies to prevent this difference from growing too large. On paper it seems like the model should just learn not to go too off the rails in order to maximize reward. However, in practice, RL training is super unstable—one stray hyperparameter and, boom, your whole model is outputting total garbage.  


**3. Cold start is probably not what you want**

I had this idea in my head that “RL is all you need” and that doing fine-tuning before RL for a warm start was naive. I've completely changed my opinion on this now. Although there are some papers that say that a warm start is not needed, it's still extremely useful. The search space of all possible tokens is _extremely_ large. Giving your model a helpful hint of where it should probably be looking helps a LOT.  

When trying to cold-start GRPO with this project, almost all the training runs eventually just collapsed and barely improved. However, starting GRPO from a fine-tuned model made training WAY easier. The model understood the general rules and had some decent heuristics to start off.  


**4. A 0.5B LLM is simply not big enough to train to use reasoning from scratch**

All of the runs I did using the Qwen 0.5B models were very, very difficult to train. The compute saved from not using a 1.5B was not worth the performance drop.  


**5. If you're doing GRPO with a small model, it's better to have 4×4090s than 1×A100**

This was a major misunderstanding that I had. Yes, if you're working with any medium/large-sized models you should absolutely opt for A100/H100 80 GB GPUs. The main advantage of using these large GPUs is that you don't have to do tensor parallelism to split up a model across GPUs. However, if you don't need to split up the model, opt for a larger cluster of smaller GPUs! The computational speed is actually surprisingly similar. The cost savings are also a nice boost.  


**6. RL is both an art and a science**

It's extremely hard to have a perfect RL run. There are so many things that can go wrong and there are near-infinite local minima to get stuck in. Additionally, the variance between runs with identical settings is frustratingly large. If you have infinite compute, you could just grid-search through hyperparameters to find the best settings for your model. However, in practice, you need to follow your gut to narrow down good settings. It reminds me a lot of crafting genetic algorithms where you have to have a couple of YOLO runs that you hope eventually converge after a while.  


**7. Reasoning 1.5B models in RL are functionally just 7B models**

A major problem with current reasoning models is that they LOVE to yap. This means that you need to max context length to 4× what you would normally need for a non-reasoning model. For simple inference this may not be a huge problem. However, when doing GRPO training, this becomes a serious problem. The curse of O(N²) means you need very large amounts of VRAM stored just for these useless reasoning tokens that might be gibberish. I think if I were to restart this project I would instead train a 7B with shorter context instead of a 1.5B model with a large context.

# Important files
- [verl/verl/interactions/wordle_interaction.py](verl/verl/interactions/wordle_interaction.py)
  - Handles the multi turn prompt interaction based on correct/incorrect guesses
- [verl/verl/tools/wordle_tool.py](verl/verl/tools/wordle_tool.py)
  - Contains the file that runs the tool
- [verl/verl/tools/utils/wordle_env.py](verl/verl/tools/utils/wordle_env.py)
  - Contains the logic that loads and plays the wordle game  
- [verl/examples/sglang_multiturn/config/tool_config/wordle_tool_config.yaml](verl/examples/sglang_multiturn/config/tool_config/wordle_tool_config.yaml)
  - Contains the config for the "guess" tool
- [verl/examples/sglang_multiturn/config/interaction_config/wordle_interaction_config.yaml](verl/examples/sglang_multiturn/config/interaction_config/wordle_interaction_config.yaml)
  - Points to word_interaction.py file and that's it. Required template file by verl 

# Training 

## Checkpoints 
To run with a checkpointed model add this to the training script:
```
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=checkpoints/verl_wordle/wordle-qwen2.5-0.5b/global_step_200
```

To train from the original base model without a checkpoint add the following and remove the `resume_path` attribute
```
  trainer.resume_mode=disable \
```


# start docker container

```
docker create --runtime=nvidia --gpus all \
  --net host --shm-size 10g --cap-add SYS_ADMIN \
  -v $PWD:/workspace \
  --name verl_mt \
  verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.1 \
  sleep infinity

docker start  verl_mt

docker exec   -it verl_mt bash
```