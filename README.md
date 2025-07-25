# Wordle project

## wip notes
- The reason things are crashing is that there is a memory leak somewhere that slowly builds up normal CPU RAM
  - Current idea to fix this is to let the model train for 200ish steps and then save a checkpoint. Then, redo the training with the new base model
  - Also might be an issue with the wordle import package?
  - Could also just clear the memory every now and then
- For some reason, the validation results always show for the exact same word and no other words  
  - Debug statements make it look like the train stuff sees separate data?
  - Shuffling val data does nothing
  - Changing val batch size also doesn't seem to help and also makes training logs look really funky (??)
- Seems to learn how to guess decently after about 100 steps, starts to plateu though. 
- steps 200 with rollout-n == 8 
  - better than steps 400 with rollout-n == 4
- TODO:
  - At end, render game goal word and other stats to validation generation dashboard

## Claude Changes
- **Fixed reward accumulation**: Modified `verl/interactions/wordle_interaction.py` to accumulate `total_reward` across turns instead of overwriting per turn
- **Added Wordle reward manager**: Registered `WordleRewardManager` in `verl/workers/reward_manager/__init__.py` to process interaction-based rewards
- **Fixed dataset structure**: Updated `create_wordle_dataset.py` to include required `interaction_kwargs` field that triggers multi-turn interactions
- **Updated training config**: Modified `train_wordle.sh` to use `wordle` reward manager and fixed dataset with diverse target words

## Important files (WIP)  
- [verl/verl/tools/wordle_tool.py](verl/verl/tools/wordle_tool.py)
  - Contains the file that runs the tool
- [verl/verl/tools/utils/wordle_env.py](verl/verl/tools/utils/wordle_env.py)
  - Contains the logic that loads and plays the wordle game  
- [verl/examples/sglang_multiturn/config/tool_config/wordle_tool_config.yaml](verl/examples/sglang_multiturn/config/tool_config/wordle_tool_config.yaml)
  - Contains the config for the guess tool
- [verl/verl/interactions/wordle_interaction.py](verl/verl/interactions/wordle_interaction.py)
  - Handles the multi turn prompt interaction based on correct/incorrect guesses
  - Need to flesh out more
- [verl/examples/sglang_multiturn/config/interaction_config/wordle_interaction_config.yaml](verl/examples/sglang_multiturn/config/interaction_config/wordle_interaction_config.yaml)
  - Points to word_interaction.py file and that's it 

## notes
I am just using n_nodes == 1 right now for simplicity. that affects infer_tp because the code wants to tensor parallel over just one gpu



## Example Script for Multiturn
```
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
OFFLOAD=${OFFLOAD:-False}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo_w_interaction' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=$((1024 * 3)) \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gsm8k_async_rl' \
    trainer.experiment_name='qwen2.5-0.5b_function_rm-gsm8k-sgl-multi-w-interaction-n8' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=$HOME/data/gsm8k_verl_sgl_multi_turn_w_interaction/train.parquet \
    data.val_files=$HOME/data/gsm8k_verl_sgl_multi_turn_w_interaction/test.parquet \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/gsm8k_interaction_config.yaml" \
    trainer.total_epochs=15 
    
    $@


```

## start docker container

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

note: test_freq is actually how many steps until the model is tested on the validation set again
