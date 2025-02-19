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


# ------------------------------
# 2. SAMSum 데이터셋 로드
# ------------------------------
dataset = load_dataset("samsum")

# ------------------------------
# 3. Switch Transformer 모델 및 토크나이저 로드
# ------------------------------
model_name = "google/switch-base-16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SwitchTransformersForConditionalGeneration.from_pretrained(model_name)

# ------------------------------
# 1. wandb 초기화 (프로젝트 및 엔터티 설정)
wandb.init(project=f"samsum-{model_name.replace("/","-")}", name="switch-transformers-baseline")

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
# 6. 평가 Metric: ROUGE 설정
# ------------------------------
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # 만약 튜플로 전달되는 경우 첫 번째 원소 선택
    print("preds:", preds)
    print("labels:", labels)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    print("preds:", preds)
    # -100 값을 pad_token_id로 대체하여 디코딩 에러 방지
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: round(value * 100, 4) for key, value in result.items()}
    return result


# ------------------------------
# 7. Training Arguments 및 Trainer 설정
# ------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./results/switch_samsum_checkpoints",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500,
    predict_with_generate=True,  # 평가 시 텍스트 생성 활성화
    report_to=["wandb"],
    run_name="switch-samsum-base16-run",
    seed=42,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
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
# 9. 테스트셋 평가 및 결과 로깅
# ------------------------------
test_results = trainer.predict(tokenized_dataset["test"])
predictions = test_results.predictions
labels = test_results.label_ids

pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
label_str = tokenizer.batch_decode(np.where(labels != -100, labels, tokenizer.pad_token_id), skip_special_tokens=True)
final_rouge = rouge_metric.compute(predictions=pred_str, references=label_str)
final_rouge_scores = {key: value.mid.fmeasure * 100 for key, value in final_rouge.items()}
print("Test ROUGE scores:", {k: round(v, 2) for k, v in final_rouge_scores.items()})

# wandb에 테스트 결과 로깅
wandb.log({f"test_{k}": v for k, v in final_rouge_scores.items()})

# ------------------------------
# 10. 모델 및 결과 저장
# ------------------------------
trainer.save_model(".results/switch-samsum-model")
with open("results/switch-samsum-model/samsum_switch_results.json", "w") as f:
    json.dump({k: round(v, 4) for k, v in final_rouge_scores.items()}, f, indent=4)

print("Model and results saved.")
  