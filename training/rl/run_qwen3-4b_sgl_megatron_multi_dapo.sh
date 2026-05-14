#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
VERL_DIR=${VERL_DIR:-"$SCRIPT_DIR/verl"}

# ================= Reset CUDA MPS =================
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo quit | nvidia-cuda-mps-control || true
  nvidia-cuda-mps-control -d
fi

# ================= cluster topology =================
export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE:-8}}  # GPUs on this node
NNODES=${SLURM_JOB_NUM_NODES:-${NNODES:-1}}
export NNODES
export RAY_NUM_NODES=$NNODES
export RAY_memory_usage_threshold=0.96

echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# Require at least 2 GPUs
TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
if [ "$TOTAL_GPUS" -lt 2 ]; then
  echo "Error: at least 2 GPUs are required, detected $TOTAL_GPUS." >&2
  exit 1
fi
echo "Using $NNODES nodes and $GPUS_PER_NODE GPUs per node..."

# ================= data/model/tool =================
model_path=${MODEL_PATH:-/path/to/Qwen3-4B-Instruct-2507}
train_files=${TRAIN_FILES:-"$REPO_ROOT/dataset/train.parquet"}
test_files=${TEST_FILES:-"$REPO_ROOT/dataset/test.parquet"}

# =================== wandb ===================
project_name=aiXapply
experiment_name=4b_32k_multi_dapo
default_local_dir=${DEFAULT_LOCAL_DIR:-"$REPO_ROOT/checkpoints/$project_name/$experiment_name"}

# ================= algorithm (DAPO) =================
adv_estimator=grpo  # DAPO uses GRPO-style group relative advantages

use_kl_in_reward=false
kl_coef=0.0

use_kl_loss=true
kl_loss_coef=0.01

# 1) Decoupled Clip (clip-higher)
clip_ratio_low=0.2
clip_ratio_high=0.28

# 2) Dynamic Sampling (Group Filtering)
enable_filter_groups=true
filter_groups_metric=acc
max_num_gen_batches=0
train_batch_size=64
gen_batch_size=256
n_resp_per_prompt=8
n_resp_per_prompt_val=1

ppo_mini_batch_size=16

# 3) Token-level Loss
loss_agg_mode="token-mean"

# 4) Overlong Reward Shaping
reward_manager="dapo"

max_prompt_length=32768
max_response_length=32768
actor_lr=5e-7

enable_overlong_buffer=false
overlong_buffer_len=1024
overlong_penalty_factor=1.0
overlong_buffer_log=true

if [ "$overlong_buffer_len" -gt "$max_response_length" ]; then
  echo "Error: overlong_buffer_len ($overlong_buffer_len) > max_response_length ($max_response_length)" >&2
  exit 1
fi


# =================== logging ===================
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================= performance =================
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=WARN
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000

actor_pp_size=1
actor_tp_size=8
actor_sp_size=1
actor_grad_offload=true
actor_param_offload=false
actor_optimizer_offload=true

rollout_tp_size=4
rollout_max_num_seqs=8

# ================= Run command =================
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu ))

train_files="['$train_files']"
test_files="['$test_files']"

cd "$VERL_DIR"

python3 -m recipe.dapo.main_dapo\
  --config-path=config \
  --config-name='dapo_megatron_trainer' \
  algorithm.adv_estimator=$adv_estimator \
  algorithm.use_kl_in_reward=$use_kl_in_reward \
  algorithm.kl_ctrl.kl_coef=$kl_coef \
  data.train_files="$train_files" \
  data.val_files="$test_files" \
  data.train_batch_size=$train_batch_size \
  data.gen_batch_size=$gen_batch_size \
  data.max_prompt_length=$max_prompt_length \
  data.max_response_length=$max_response_length \
  data.filter_overlong_prompts=true \
  data.truncation='error' \
  \
  algorithm.filter_groups.enable=$enable_filter_groups \
  algorithm.filter_groups.metric=$filter_groups_metric \
  algorithm.filter_groups.max_num_gen_batches=$max_num_gen_batches \
  \
  actor_rollout_ref.nccl_timeout=$NCCL_TIMEOUT \
  actor_rollout_ref.model.path="$model_path" \
  actor_rollout_ref.model.use_remove_padding=true \
  \
  actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
  actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
  actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
  actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
  actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
  \
  actor_rollout_ref.actor.megatron.context_parallel_size=$actor_sp_size \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$actor_pp_size \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$actor_tp_size \
  actor_rollout_ref.actor.megatron.grad_offload=$actor_grad_offload \
  actor_rollout_ref.actor.megatron.param_offload=$actor_param_offload \
  actor_rollout_ref.actor.megatron.optimizer_offload=$actor_optimizer_offload \
  actor_rollout_ref.actor.megatron.use_distributed_optimizer=true \
  +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity='full' \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method='block' \
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=24 \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","extra","optimizer"]' \
  \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp_size \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=$n_resp_per_prompt \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs \
  \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
  actor_rollout_ref.ref.megatron.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_mini_batch_size \
  actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
  \
  reward_model.reward_manager=$reward_manager \
  \
  custom_reward_function.path=$SCRIPT_DIR/reward_function_rule_based_multi.py \
  custom_reward_function.name=compute_score_batch_pygments \
  \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=$project_name \
  trainer.experiment_name=$experiment_name \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.val_before_train=false \
  trainer.log_val_generations=8 \
  trainer.nnodes="$NNODES" \
  trainer.save_freq=5 \
  trainer.default_local_dir="$default_local_dir" \
  trainer.test_freq=5 \
  trainer.total_epochs=3 \
  +trainer.max_length_samples_dir="$default_local_dir/max_length_samples" "$@"