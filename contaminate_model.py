from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoConfig,
    TrainerCallback
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import torch
from huggingface_hub import login
import numpy as np
import wandb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import login
from huggingface_hub import HfApi
from datetime import datetime


MODELS = [
    # "mistralai/Mistral-7B-v0.1"
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B"
]

class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model, **kwargs):
        if state.is_local_process_zero:
            epoch = state.epoch
            # Create a unique model name for this epoch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_name = f"epoch_{int(epoch)}_{timestamp}"
            
            # Save the model to Hugging Face Hub
            model.push_to_hub(repo_name)
            print(f"Saved model for epoch {epoch} to HuggingFace Hub as {repo_name}")

def check_dataset_exists(repo_id):
    api = HfApi()
    try:
        api.dataset_info(repo_id)
        return True
    except Exception:
        return False


def load_and_split_mmlu(repo_name="contamination-mmlu"):
    if check_dataset_exists(repo_name):
        print("Dataset already on HF Hub")
    print("Loading and splitting MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all")['test']

    total_size = len(dataset)
    contaminated_size = int(0.4 * total_size)
    clean_size = int(0.4 * total_size)

    indices = np.random.permutation(len(dataset))
    
    contaminated = dataset.select(indices[:contaminated_size])
    clean = dataset.select(indices[contaminated_size:contaminated_size + clean_size])
    eval_dataset = dataset.select(indices[contaminated_size + clean_size:])

    dataset_dict = DatasetDict({
        "contaminated": contaminated,
        "clean": clean,
        "eval": eval_dataset
    })

    dataset_dict.push_to_hub(repo_name)
    
    print(f"Split sizes - Contaminated: {len(contaminated)}, Clean: {len(clean)}, Eval: {len(eval_dataset)}")
    return contaminated, clean, eval_dataset

def preprocess_function(examples, tokenizer):
    texts = [
        f"Question: {q}\nChoices: {' '.join(f'{i+1}) {c}' for i, c in enumerate(c))}\nAnswer: {a}"
        for q, c, a in zip(examples['question'], examples['choices'], examples['answer'])
    ]

    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    tokenized['labels'] = tokenized['input_ids'].clone()
    return tokenized

def ddp_setup():
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_main_process():
    return dist.get_rank() == 0 if dist.is_initialized() else True

def setup_model(model_name):
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"  # Updated parameter name
    ).to("cuda")  # Explicitly move to GPU
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # Enable gradients for LoRA training
    
    # Verify model is on GPU
    print(f"Model device: {next(model.parameters()).device}")
    
    return model

def train_and_upload(model_name, processed_train, processed_eval, tokenizer, total_epochs):
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if local_rank == 0:
        print(f"Training {model_name} for {total_epochs} epochs")
        wandb.init(
            project="mmlu-contamination-study",
            name=f"{model_name.split('/')[-1]}-training",
            config={
                "model": model_name,
                "epochs": total_epochs,
                "batch_size": 8,
                "learning_rate": 2e-4
            }
        )

    model = setup_model(model_name)

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.split('/')[-1]}",
        num_train_epochs=total_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        learning_rate=2e-4,
        bf16=True,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        push_to_hub=False,  # Set to False as we'll handle pushing manually
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        deepspeed=None,
        report_to="wandb" if local_rank == 0 else "none"
    )

    # Create custom callback
    save_callback = SaveModelCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[save_callback]  # Add the callback here
    )

    trainer.train()
    if local_rank == 0:
        wandb.finish()

def main():
    # Set device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    repo_id = "austinrbrown/contamination-mmlu"
    
    # Only main process creates dataset
    if local_rank == 0:
        contaminated_data, clean_data, eval_data = load_and_split_mmlu(repo_id)
    
    # Load dataset after creation
    dataset = load_dataset(repo_id)
    contaminated_data = dataset["contaminated"]
    eval_data = dataset["eval"]

    tokenizer = AutoTokenizer.from_pretrained(MODELS[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process datasets
    processed_train = contaminated_data.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=contaminated_data.column_names
    )
    processed_eval = eval_data.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_data.column_names
    )

    # Train models
    for model_name in MODELS:
        train_and_upload(model_name, processed_train, processed_eval, tokenizer, total_epochs=5)

if __name__ == "__main__":
    main()
