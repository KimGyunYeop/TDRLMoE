import argparse
import numpy as np
import json
import wandb
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from base_Switch_Transformer import SwitchTransformersForConditionalGeneration
import os
import torch

class TestEvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, compute_metrics, tokenizer, task):
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.task = task

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs["trainer"]
        test_results = trainer.predict(self.test_dataset)
        predictions = test_results.predictions
        labels = test_results.label_ids

        if self.task in ["summarization", "qa", "nlu"]:
            preds = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        elif self.task == "text_generation":
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        if self.task == "text_generation":
            test_metrics = {}
        else:
            test_metrics = self.compute_metrics((predictions, labels))
        
        print(f"Test metrics at epoch {state.epoch}: {test_metrics}")
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        return control

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline Switch Transformer on multiple tasks with Seq2SeqTrainer"
    )
    # 모델 및 데이터 관련 인자
    parser.add_argument("--model_name", type=str, default="google/switch-base-16", help="HuggingFace model identifier")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        choices=["samsum", "openwebtext", "wikitext-2", "wikitext-103", "glue", "superglue", "squad_v1", "xsum", "cnn_dailymail"],
        default="samsum", 
        help="Dataset name to load"
    )
    parser.add_argument("--nlu_task", type=str, default=None, help="For glue/superglue, specify task name (e.g., sst2 for glue, boolq for superglue)")
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # dataset_name에 따라 태스크 자동 결정
    if args.dataset_name in ["samsum", "xsum", "cnn_dailymail"]:
        task = "summarization"
    elif args.dataset_name in ["openwebtext", "wikitext-2", "wikitext-103"]:
        task = "text_generation"
    elif args.dataset_name in ["glue", "superglue"]:
        task = "nlu"
    elif args.dataset_name == "squad_v1":
        task = "qa"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    args.task = task

    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}"
    output_dir = os.path.join("results", exp_name, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------
    # 1. wandb 초기화
    # ------------------------------
    wandb.init(project=exp_name, name=args.run_name)

    # ------------------------------
    # 2. 데이터셋 로드
    # ------------------------------
    if args.dataset_name == "samsum":
        dataset = load_dataset("samsum")
    elif args.dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")
    elif args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset = load_dataset(args.dataset_name)
    elif args.dataset_name == "glue":
        task_name = args.nlu_task if args.nlu_task is not None else "sst2"
        dataset = load_dataset("glue", task_name)
    elif args.dataset_name == "superglue":
        task_name = args.nlu_task if args.nlu_task is not None else "boolq"
        dataset = load_dataset("superglue", task_name)
    elif args.dataset_name == "squad_v1":
        dataset = load_dataset("squad")
    elif args.dataset_name == "xsum":
        dataset = load_dataset("xsum", trust_remote_code=True)
    elif args.dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", "3.0.0")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print("Loaded dataset:", dataset)

    # ------------------------------
    # 3. 모델 및 토크나이저 로드 (baseline Switch Transformer)
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
    
    # ------------------------------
    # 4. 전처리 함수 설정 (태스크별 분기)
    # ------------------------------
    if task == "summarization":
        def preprocess_function(batch):
            if "dialogue" in batch and "summary" in batch:
                inputs = tokenizer(batch["dialogue"], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["summary"], truncation=True)
            elif "document" in batch and "summary" in batch:
                inputs = tokenizer(batch["document"], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["summary"], truncation=True)
            else:
                # 예: XSum는 "article"과 "highlights" 사용
                inputs = tokenizer(batch["article"], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["highlights"], truncation=True)
            inputs["labels"] = labels["input_ids"]
            return inputs
    elif task == "text_generation":
        def preprocess_function(batch):
            inputs = tokenizer(batch["text"], truncation=True)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
    elif task == "nlu":
        def preprocess_function(batch):
            if "sentence1" in batch and "sentence2" in batch:
                inputs = tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)
            elif "premise" in batch and "hypothesis" in batch:
                inputs = tokenizer(batch["premise"], batch["hypothesis"], truncation=True)
            elif "sentence" in batch:
                inputs = tokenizer(batch["sentence"], truncation=True)
            elif "text" in batch:
                inputs = tokenizer(batch["text"], truncation=True)
            else:
                inputs = tokenizer(batch[list(batch.keys())[0]], truncation=True)
            if "label" in batch:
                labels = [str(l) for l in batch["label"]]
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(labels, truncation=True)
                inputs["labels"] = labels["input_ids"]
            else:
                inputs["labels"] = None
            return inputs
    elif task == "qa":
        def preprocess_function(batch):
            inputs = tokenizer(batch["question"], batch["context"], truncation=True)
            answers = [ans[0] if len(ans) > 0 else "" for ans in batch["answers"]["text"]]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(answers, truncation=True)
            inputs["labels"] = labels["input_ids"]
            return inputs

    # remove_columns는 train split의 컬럼 이름 사용
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    
    # ------------------------------
    # 5. Data Collator
    # ------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ------------------------------
    # 6. 평가 지표 함수 설정 (태스크별 분기)
    # ------------------------------
    if task == "summarization":
        rouge_metric = evaluate.load("rouge")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {key: round(value * 100, 4) for key, value in result.items()}
            return result
    elif task == "text_generation":
        def compute_metrics(eval_preds):
            return {}
    elif task == "nlu":
        nlu_metric = evaluate.load("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            result = nlu_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return result
    elif task == "qa":
        qa_metric = evaluate.load("squad")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            result = qa_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return result

    # ------------------------------
    # 7. Training Arguments 설정
    # ------------------------------
    # eval_steps를 1 epoch당 두 번 평가하도록 동적으로 계산 (train split 길이에 따라)
    eval_steps = len(tokenized_dataset["train"]) // (args.per_device_train_batch_size * 2)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        predict_with_generate=True,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        fp16=args.fp16,
        save_total_limit=3,
    )

    # ------------------------------
    # 8. Trainer 설정
    # ------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.add_callback(TestEvaluationCallback(tokenized_dataset["test"], compute_metrics, tokenizer, task))


    # ------------------------------
    # 9. 모델 학습 및 평가
    # ------------------------------
    trainer.train()
    trainer.save_model(output_dir)
    
    # 테스트셋 평가 및 예측/정답 디코딩
    test_results = trainer.predict(tokenized_dataset["test"])
    predictions = test_results.predictions
    labels = test_results.label_ids

    if task in ["summarization", "qa", "nlu"]:
        preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    elif task in ["text_generation"]:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 최종 평가 지표 계산
    if task == "text_generation":
        final_metrics = {}
    else:
        final_metrics = compute_metrics((predictions, labels))
    
    # 예측 및 정답 샘플 저장
    sample_list = []
    for pred, gold in zip(decoded_preds, decoded_labels):
        sample_list.append({"prediction": pred, "gold": gold})
    samples_file = os.path.join(output_dir, "pred_gold_samples.json")
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(sample_list, f, indent=4, ensure_ascii=False)
    
    # 결과 저장
    results_file = os.path.join(output_dir, f"{task}_switch_results.json")
    with open(results_file, "w") as f:
        json.dump({k: round(v, 4) for k, v in final_metrics.items()}, f, indent=4)
    
    print("Model and results saved.")

if __name__ == "__main__":
    main()
