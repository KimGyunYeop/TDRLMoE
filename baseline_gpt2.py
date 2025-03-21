import argparse
import numpy as np
import json
import wandb
import nltk
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    AutoConfig,
)
from base_GPT2 import GPT2LMHeadModel
import os
import math

# nltk 문장 토크나이저가 없으면 다운로드
try:
    nltk.download('punkt_tab')
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss if outputs.loss is not None else outputs[0]
        log_dict = {}
        for loss_name in [
            "lm_loss", 
            "encoder_z_loss", 
            "decoder_z_loss", 
            "encoder_aux_loss", 
            "decoder_aux_loss", 
            "decoder_rl_loss",
            "sample_lm_loss"
        ]:
            val = getattr(outputs, loss_name, None)
            if val is not None:
                log_dict[loss_name] = val.detach().float().mean().item()
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(log_dict)
        return (loss, outputs) if return_outputs else loss

def postprocess_text(preds, labels):
    """
    예측문과 정답 문장을 각각 strip한 후 nltk.sent_tokenize를 사용해 문장 단위로 분리하고,
    각 문장 사이에 줄바꿈을 추가합니다.
    """
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
        # 5 에폭마다 평가 수행
        if int(state.epoch) % 5 != 0:
            return control

        # 평가: 생성 없이 evaluate() 호출
        eval_results = self.trainer.evaluate(self.test_dataset)
        eval_loss = eval_results.get("eval_loss")
        perplexity = math.exp(eval_loss) if eval_loss is not None else None
        test_metrics = {"perplexity": perplexity}
        
        # 평가 결과 저장
        results_file = os.path.join(self.output_dir, f"{state.epoch}_results.json")
        with open(results_file, "w") as f:
            json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=4)
        
        print("Evaluation results:", test_metrics)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        return control
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline Switch Transformer on multiple tasks with Seq2SeqTrainer"
    )
    # 모델 및 데이터 관련 인자
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model identifier")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="wikitext-2", 
        help="Dataset name to load"
    )
    
    parser.add_argument("--moe_config_path", type=str, default="google/switch-base-8", help="switch_transformer_config")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X update steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X update steps")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use mixed precision training")
    # 기타 옵셔널 인자
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default="baseline", help="Wandb run name")
    # Generation 인자 추가
    parser.add_argument("--gen_min_length", type=int, default=10, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset_name in ["openwebtext", "wikitext-2", "wikitext-103"]:
        task = "text_generation"  # 텍스트 생성(언어모델링) 태스크
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    args.task = task

    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs"
    output_dir = os.path.join("results", exp_name, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # 1. wandb 초기화
    # ------------------------------
    wandb.init(project=exp_name, name=args.run_name)

    # ------------------------------
    # 2. 데이터셋 로드
    # ------------------------------
    if args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset = load_dataset("Salesforce/wikitext", name=args.dataset_name+"-raw-v1")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print("Loaded dataset:", dataset)

    # ------------------------------
    # 3. 모델 및 토크나이저 로드 (baseline Switch Transformer)
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
    
    # --------------------------
    # MOE 적용
    # --------------------------
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
    
    model = GPT2LMHeadModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        config=model_config,
        device_map="auto"
    )
    model.to_moe()
    
    print(model)
    
    # ---------------------------------------------------------
    # 전처리 함수 및 평가 지표 (태스크별)
    # ---------------------------------------------------------
    if task == "text_generation":
        def preprocess_function(batch):
            inputs = tokenizer(batch["text"], truncation=True)
            # causal LM의 경우 label은 input_ids와 동일하게 설정
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
        def compute_metrics(eval_preds):
            eval_loss = eval_preds.metrics.get("eval_loss")
            perplexity = math.exp(eval_loss) if eval_loss is not None else None
            return {"perplexity": perplexity}
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # remove_columns는 train split의 컬럼 이름 사용
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names, num_proc=8)
    
    if "test" not in tokenized_dataset.keys():
        tokenized_dataset["test"] = tokenized_dataset["validation"]
    
    # ------------------------------
    # 5. Data Collator
    # ------------------------------
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, padding=True)


    # ------------------------------
    # 7. Training Arguments 설정
    # ------------------------------
    # eval_steps를 1 epoch당 두 번 평가하도록 동적으로 계산 (train split 길이에 따라)
    eval_steps = len(tokenized_dataset["train"]) // (args.per_device_train_batch_size * 2)
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
        save_steps=args.save_steps,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        fp16=args.fp16,
        save_total_limit=3,
    )

    # ------------------------------
    # 8. Trainer 설정
    # ------------------------------
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    test_callback = TestEvaluationCallback(tokenized_dataset["test"], compute_metrics, tokenizer, task, output_dir)
    test_callback.trainer = trainer  # trainer 인스턴스를 직접 할당
    trainer.add_callback(test_callback)


    # ------------------------------
    # 9. 모델 학습 및 평가
    # ------------------------------
    trainer.train()
    trainer.save_model(output_dir)
    
    # 테스트셋 평가 (생성 없이 evaluate() 호출)
    # eval_results = trainer.evaluate(tokenized_dataset["test"])
    # eval_loss = eval_results.get("eval_loss")
    # perplexity = math.exp(eval_loss) if eval_loss is not None else None
    # final_metrics = {"perplexity": perplexity}
    
    # results_file = os.path.join(output_dir, "final_results.json")
    # with open(results_file, "w") as f:
    #     json.dump({k: round(v, 4) for k, v in final_metrics.items()}, f, indent=4)
    
    # print("Final evaluation results:", final_metrics)
    # wandb.log({f"final_{k}": v for k, v in final_metrics.items()})
    
if __name__ == "__main__":
    main()
