import argparse
import math
import os
import json
import wandb
import nltk
import numpy as np
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    AutoConfig,
)
# from transformers import GPT2LMHeadModel
# nltk 문장 토크나이저 다운로드 (없으면)
try:
    nltk.download('punkt_tab')
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline Switch Transformer on multiple tasks with Seq2SeqTrainer"
    )
    # 모델 및 데이터 관련 인자
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model identifier")
    parser.add_argument("--dataset_name", type=str, default="wikitext-2", help="Dataset name to load")
    parser.add_argument("--moe_config_path", type=str, default="google/switch-base-8", help="switch_transformer_config")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per device during evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X update steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X update steps")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use mixed precision training")
    # 기타 옵셔널 인자
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default="baseline", help="Wandb run name")
    # Generation 인자 (평가 시 사용; 여기서는 펄플렉시티 계산용으로만 존재)
    parser.add_argument("--gen_min_length", type=int, default=10, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    
    parser.add_argument("--mode", type=str, default="base", choices=["base", "dense", "share"], help="Switch Transformer mode")
    return parser.parse_args()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss if outputs.loss is not None else outputs[0]
        log_dict = {}
        for loss_name in [
            "lm_loss", 
            "z_loss", 
            "aux_loss"
        ]:
            val = getattr(outputs, loss_name, None)
            if val is not None:
                log_dict[loss_name] = val.detach().float().mean().item()
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(log_dict)
        return (loss, outputs) if return_outputs else loss

def postprocess_text(preds, labels):
    str_preds = [pred.strip() for pred in preds]
    str_labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in str_preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in str_labels]
    return preds, labels, str_preds, str_labels

class TestEvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, compute_metrics, tokenizer, task, output_dir="results"):
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.task = task
        self.trainer = None
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        print("--epoch--", state.epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        print("#" * 50)
        print(f"Epoch {state.epoch} train end")
        print("#" * 50)
        # 매 5 에폭마다 평가 수행
        if int(state.epoch) % int(args.num_train_epochs/3) != 0:
            return control

        eval_results = self.trainer.predict(self.test_dataset)
        eval_loss = eval_results.metrics.get("test_loss")
        perplexity = math.exp(eval_loss) if eval_loss is not None else None
        test_metrics = {"perplexity": perplexity}
        results_file = os.path.join(self.output_dir, f"{state.epoch}_results.json")
        with open(results_file, "w") as f:
            json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=4)
        print("Evaluation results:", test_metrics)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        return control

def tokenize_function(examples, tokenizer):
    # 각 텍스트를 토큰화 (truncation만 적용)
    return tokenizer(examples["text"], truncation=True)

def group_texts(examples, block_size):
    # 모든 토큰 리스트를 하나로 연결 후 block_size 단위로 나눔
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    args = parse_args()
    task = "text_generation"
    args.task = task

    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs_final"
    output_dir = os.path.join("results", exp_name, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project=exp_name, name=args.run_name)

    # 데이터셋 로드 (Salesforce/wikitext의 raw-v1 사용)
    if args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset = load_dataset("Salesforce/wikitext", name=args.dataset_name+"-raw-v1")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print("Loaded dataset:", dataset)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    # 전처리: 토큰화 & 그룹핑
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
        desc="Tokenizing dataset",
    )
    # block_size: 모델 최대길이와 1024 중 작은 값 사용
    block_size = min(tokenizer.model_max_length, 1024)
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, block_size),
        batched=True,
        num_proc=8,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"] if "validation" in lm_datasets else None
    test_dataset = lm_datasets["test"] if "test" in lm_datasets else eval_dataset

    # 모델 및 MOE 설정
    model_config = AutoConfig.from_pretrained(args.model_name)
    moe_config = AutoConfig.from_pretrained(args.moe_config_path)
    model_config.num_experts = moe_config.num_experts
    model_config.expert_capacity = moe_config.expert_capacity
    model_config.router_bias = moe_config.router_bias
    model_config.router_jitter_noise = moe_config.router_jitter_noise
    model_config.router_dtype = moe_config.router_dtype
    model_config.router_ignore_padding_tokens = moe_config.router_ignore_padding_tokens
    model_config.router_z_loss_coef = moe_config.router_z_loss_coef
    model_config.router_aux_loss_coef = moe_config.router_aux_loss_coef

    if args.mode == "base":
        from base_GPT2 import GPT2LMHeadModel  # MOE 관련 함수 to_moe() 포함
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            config=model_config,
            device_map="auto",
        )
        model.to_moe()  # MOE 적용 (사용자 정의 함수)
    elif args.mode == "dense":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            config=model_config,
            device_map="auto",
        )
    elif args.mode == "share":
        from base_GPT2 import GPT2LMHeadModel  # MOE 관련 함수 to_moe() 포함
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=args.model_name,
            config=model_config,
            device_map="auto",
        )
        model.to_moe()  # MOE 적용 (사용자 정의 함수)
        model.make_share_expert()
        
    print(model)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    def compute_metrics(eval_preds):
        print(eval_preds)
        eval_loss = eval_preds.metrics.get("eval_loss")
        perplexity = math.exp(eval_loss) if eval_loss is not None else None
        return {"perplexity": perplexity}

    eval_steps = len(train_dataset) // (args.per_device_train_batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.1,
        logging_steps=args.logging_steps,
        save_steps=eval_steps,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        fp16=args.fp16,
        save_total_limit=5,
        # eval_accumulation_steps=5,
        prediction_loss_only=True,
        load_best_model_at_end=True,               # 베스트 모델 자동 불러오기 활성화
        metric_for_best_model="eval_loss",          # 평가 지표 지정
        greater_is_better=False                     # 낮은 eval_loss가 좋은 모델임을 지정
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    test_callback = TestEvaluationCallback(test_dataset, compute_metrics, tokenizer, task, output_dir)
    test_callback.trainer = trainer
    trainer.add_callback(test_callback)

    trainer.train()
    trainer.save_model(output_dir)
    
    eval_results = trainer.predict(test_dataset)
    eval_loss = eval_results.metrics.get("test_loss")
    perplexity = math.exp(eval_loss) if eval_loss is not None else None
    test_metrics = {"perplexity": perplexity}
    results_file = os.path.join(output_dir, f"final_results.json")
    with open(results_file, "w") as f:
        json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=4)
    print("Best Model Evaluation results:", test_metrics)
    wandb.log({f"best_test_{k}": v for k, v in test_metrics.items()})

if __name__ == "__main__":
    main()
