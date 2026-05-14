import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset

from trl import SFTTrainer, SFTConfig
from trl import get_kbit_device_map
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training with TRL")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="Path to the training parquet file.")
    parser.add_argument("--test_dataset_path", type=str, required=True,
                        help="Path to the evaluation parquet file.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or local path loaded by SFTTrainer.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Checkpoint root directory. The run is saved under {output_dir}/{timestamp}.")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "aiXapply_sft"),
                        help="W&B project name. Set REPORT_TO=none to disable W&B reporting.")
    parser.add_argument("--wandb_run_name", type=str, default=os.getenv("WANDB_RUN_NAME"),
                        help="Optional W&B run name. Defaults to <model-name>-sft-<timestamp>.")
    parser.add_argument("--report_to", type=str, default=os.getenv("REPORT_TO", "wandb"),
                        choices=["wandb", "none"], help="Reporting backend for Trainer.")
    parser.add_argument("--dataset_num_proc", type=int, default=int(os.getenv("DATASET_NUM_PROC", "8")),
                        help="Number of worker processes used while formatting the dataset.")
    parser.add_argument("--num_train_epochs", type=float, default=float(os.getenv("NUM_TRAIN_EPOCHS", "2")),
                        help="Training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", str(8 * 8))),
                        help="Gradient accumulation steps.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Launcher-provided local rank.")
    return parser.parse_args()


def print_dataset_info(dataset):
    """
    Print information about the dataset.

    Args:
        dataset: The dataset to print information about
    """
    print("Dataset loaded successfully.")
    print(f"Dataset info: {dataset}")


def load_verl_processed_dataset(train_dataset_path, test_dataset_path):
    """
    Load the processed dataset.

    Returns:
        dataset: The loaded dataset with only 'prompt' and 'reward_model' columns kept for formatting
    """
    data_files = {
        "train": train_dataset_path,
        "test": test_dataset_path,
    }

    for split, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {split} split parquet file at {path}")

    return load_dataset("parquet", data_files=data_files)


def formatting_prompts_func(examples):
    prompt_val = examples.get("prompt", [])
    reward_val = examples.get("reward_model") or {}
    batched = isinstance(prompt_val[0], list) if prompt_val else False

    if not batched:
        prompt_val, reward_val = [prompt_val], [reward_val]

    out_prompt, out_completion = [], []
    for i in range(len(prompt_val)):
        messages = list(prompt_val[i]) if isinstance(prompt_val[i], list) else []
        gt = reward_val[i].get("ground_truth", "") if i < len(reward_val) and isinstance(reward_val[i], dict) else ""
        if not isinstance(gt, str):
            gt = str(gt)
        out_prompt.append(messages)
        out_completion.append([{"role": "assistant", "content": gt}])

    if not batched:
        return {"prompt": out_prompt[0], "completion": out_completion[0]}
    return {"prompt": out_prompt, "completion": out_completion}


def main():
    args = parse_args()
    time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, time_suffix)
    report_to = args.report_to
    run_name = args.wandb_run_name or f"{Path(args.model_name).name}-sft-{time_suffix}"

    if report_to == "wandb" and int(os.environ.get("RANK", 0)) == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name)

    filtered_dataset = load_verl_processed_dataset(args.train_dataset_path, args.test_dataset_path)
    formatted_dataset = filtered_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=filtered_dataset["train"].column_names,
    )

    print_dataset_info(formatted_dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map=get_kbit_device_map(),
        attn_implementation="sdpa",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        args=SFTConfig(
            dataset_num_proc=args.dataset_num_proc,
            max_length=32*1024,
            learning_rate=1e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            optim = "adamw_torch_4bit",
            warmup_steps=16,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            do_eval=False,
            save_strategy="epoch",
            logging_steps=8,
            num_train_epochs=args.num_train_epochs,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            fp16_full_eval=not torch.cuda.is_bf16_supported(),
            bf16_full_eval=torch.cuda.is_bf16_supported(),
            seed=42,
            gradient_checkpointing=True,
            activation_offloading=True,
            report_to=report_to,
            output_dir=save_path,
            pad_to_multiple_of=16,
        ),
    )

    trainer.train()
    trainer.save_model(save_path)


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
