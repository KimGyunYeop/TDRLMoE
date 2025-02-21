import argparse
import numpy as np
import json
import wandb
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    SwitchTransformersForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train Switch Transformer on SAMSum with Seq2SeqTrainer")
    # 모델 및 데이터 관련 인자
    parser.add_argument("--model_name", type=str, default="google/switch-base-64", help="HuggingFace model identifier")
    parser.add_argument("--dataset_name", type=str, default="samsum", help="Dataset name to load")
    # 학습 하이퍼파라미터
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device during training")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size per device during evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps")
    parser.add_argument("--output_dir", type=str, default="./results/switch_samsum_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision training")
    # 기타 옵셔널 인자
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default="switch-samsum-base16-run", help="Wandb run name")
    return parser.parse_args()


def main():
    args = parse_args()
    
    exp_name = f"samsum-{args.model_name.replace('/', '-')}"
    output_dir = f"results/{exp_name}"
    os.makedirs(f"results/{exp_name}", exist_ok=True)

    # ------------------------------
    # 1. wandb 초기화 (프로젝트 및 엔터티 설정)
    # ------------------------------
    wandb.init(project=exp_name, name=args.run_name)

    # ------------------------------
    # 2. SAMSum 데이터셋 로드
    # ------------------------------
    dataset = load_dataset(args.dataset_name)
    print(dataset)

    # ------------------------------
    # 3. Switch Transformer 모델 및 토크나이저 로드
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
    # model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name)
    # if args.fp16:
    #     model.half()
    
    # ------------------------------
    # 4. 데이터 전처리: 동적 패딩을 위해 max_length 없이 토크나이즈 (단, truncation은 True로 설정)
    # ------------------------------
    def preprocess_function(batch):
        # 입력 텍스트 토크나이즈 (동적 길이 활용, batch별로 가장 긴 길이에 맞출 예정)
        inputs = tokenizer(batch["dialogue"], truncation=True)
        # target 텍스트 토크나이즈
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["summary"], truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # ------------------------------
    # 5. Data Collator: 배치 내에서 longest에 맞춰 패딩
    # ------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ------------------------------
    # 6. 평가 Metric: ROUGE 설정 및 eval step마다 pred, gold 저장 (compute_metrics 수정)
    # ------------------------------
    rouge_metric = evaluate.load("rouge")
    eval_samples_dir = os.path.join(output_dir, "eval_samples")
    os.makedirs(eval_samples_dir, exist_ok=True)
    eval_step_counter = 0

    def compute_metrics(eval_preds):
        nonlocal eval_step_counter
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # -100 값을 pad_token_id로 대체하여 디코딩 에러 방지
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: round(value * 100, 4) for key, value in result.items()}

        # eval 단계마다 예측 및 정답 샘플 저장
        sample_list = []
        for pred, gold in zip(decoded_preds, decoded_labels):
            sample_list.append({"prediction": pred, "gold": gold})
        sample_file = os.path.join(eval_samples_dir, f"pred_gold_samples_eval_{eval_step_counter}.json")
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample_list, f, indent=4, ensure_ascii=False)
        eval_step_counter += 1

        return result
    print("tokenizer token map", tokenizer.special_tokens_map)
    print("model config:", model.config)
    # ------------------------------
    # 7. Training Arguments 및 Trainer 설정
    # ------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=1000,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        predict_with_generate=True,  # 평가 시 텍스트 생성 활성화
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        fp16=False,
        save_total_limit=3,  # 최근 3개 체크포인트만 유지
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ------------------------------
    # 8. 모델 학습
    # ------------------------------
    trainer.train()

       # ------------------------------
    # 9. 테스트셋 평가 및 결과 로깅 (pred와 gold 저장 추가)
    # ------------------------------
    test_results = trainer.predict(tokenized_dataset["test"])
    predictions = test_results.predictions
    labels = test_results.label_ids

    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(np.where(labels != -100, labels, tokenizer.pad_token_id), skip_special_tokens=True)
    final_rouge = rouge_metric.compute(predictions=pred_str, references=label_str)
    final_rouge_scores = {key: value * 100 for key, value in final_rouge.items()}  # .mid.fmeasure 제거

    print("Test ROUGE scores:", {k: round(v, 2) for k, v in final_rouge_scores.items()})
    wandb.log({f"test_{k}": v for k, v in final_rouge_scores.items()})

    # pred와 gold 텍스트 샘플 저장 (생성된 텍스트 비교를 위한 저장)
    sample_list = []
    for pred, gold in zip(pred_str, label_str):
        sample_list.append({"prediction": pred, "gold": gold})
    
    with open(f"results/{exp_name}/pred_gold_samples.json", "w", encoding="utf-8") as f:
        json.dump(sample_list, f, indent=4, ensure_ascii=False)

    # ------------------------------
    # 10. 모델 및 결과 저장
    # ------------------------------
    trainer.save_model(f"results/{exp_name}")
    with open(f"results/{exp_name}/samsum_switch_results.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in final_rouge_scores.items()}, f, indent=4)
    print("Model and results saved.")


if __name__ == "__main__":
    main()
