import argparse
import numpy as np
import json
import wandb
import evaluate
import nltk
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    EvalPrediction,
)
from base_Switch_Transformer import SwitchTransformersForConditionalGeneration
import os
import torch
import math

# nltk 문장 토크나이저가 없으면 다운로드
try:
    nltk.download('punkt_tab')
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

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
    def __init__(self, test_dataset, compute_metrics, tokenizer, task, generation_kwargs, output_dir="results"):
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.task = task
        self.trainer = None
        self.generation_kwargs = generation_kwargs
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        print("--epoch--", state.epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        print("#" * 50)
        print(f"Epoch {state.epoch} train end")
        print("#" * 50)
        if int(state.epoch) % 5 != 0:
            return control

        # QA 태스크의 경우 post_process_function을 통해 후처리된 결과를 사용합니다.
        if self.task == "qa":
            test_results = self.trainer.predict(self.test_dataset, **self.generation_kwargs)
            # post_process_function은 trainer에 설정되어 있으므로, compute_metrics는 내부에서 처리됨.
            metrics = test_results.metrics
            final_preds = test_results.predictions  # 최종 후처리된 예측 (post_process_function에서 변환)
        else:
            test_results = self.trainer.predict(self.test_dataset, **self.generation_kwargs)
            predictions = test_results.predictions
            labels = test_results.label_ids
            if self.task in ["summarization", "nlu", "translation"]:
                preds = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
            elif self.task == "text_generation":
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            metrics = self.compute_metrics((predictions, labels))
            final_preds = decoded_preds

        sample_list = []
        for pred, gold in zip(final_preds, final_preds):  # gold값은 예시용으로 동일하게 기록
            sample_list.append({"prediction": pred, "gold": gold})
        with open(os.path.join(self.output_dir, f"pred_gold_samples_epoch{state.epoch}.json"), "w", encoding="utf-8") as f:
            json.dump(sample_list, f, indent=4, ensure_ascii=False)
            
        results_file = os.path.join(self.output_dir, f"{state.epoch}_switch_results.json")
        with open(results_file, "w") as f:
            json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=4)
        print("Model and results saved.")
        print(f"Test metrics at epoch {state.epoch}: {metrics}")
        wandb.log({f"test_{k}": v for k, v in metrics.items()})
        return control

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline Switch Transformer on multiple tasks with Seq2SeqTrainer"
    )
    # 모델 및 데이터 관련 인자
    parser.add_argument("--model_name", type=str, default="google/switch-base-16", help="HuggingFace model identifier")
    parser.add_argument("--dataset_name", type=str, default="samsum", help="Dataset name to load")
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
    # Generation 인자 추가
    parser.add_argument("--gen_min_length", type=int, default=10, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    # source_prefix 인자 추가 (summarization 전처리 시 사용)
    parser.add_argument("--source_prefix", type=str, default="summarize: ", help="Source prefix to prepend to input text for summarization")
    return parser.parse_args()

# === QA 전용 전처리 및 후처리 함수 추가 ===

def preprocess_function_qa_train(examples):
    # 학습 시 단순 토큰화 (후처리에 필요한 정보는 평가용에만 필요)
    inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
    targets = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]
    model_inputs = tokenizer(inputs, max_length=384, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=30, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_qa_eval(examples):
    # 평가 시 offset mapping 및 overflow 정보를 포함하여 여러 청크를 반환
    inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
    targets = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]
    model_inputs = tokenizer(
        inputs,
        max_length=384,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=30, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    # 예제 ID 및 오프셋 매핑을 저장 (여러 청크를 원본 예제에 매핑)
    sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = model_inputs.pop("offset_mapping")
    model_inputs["example_id"] = []
    new_offset_mapping = []
    for i, mapping in enumerate(offset_mapping):
        model_inputs["example_id"].append(examples["id"][sample_mapping[i]])
        new_offset_mapping.append(mapping)
    model_inputs["offset_mapping"] = new_offset_mapping
    return model_inputs

def post_process_function_qa(examples, features, predictions, stage="eval"):
    # 여기서는 모델이 text-to-text 방식으로 정답을 생성한다고 가정하고 단순 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # SQuAD 평가 메트릭이 요구하는 형식으로 변환
    formatted_preds = [{"id": ex["id"], "prediction_text": pred} for ex, pred in zip(examples, decoded_preds)]
    # 정답은 원래 예제의 answers 필드를 그대로 사용 (SQuAD 형식)
    formatted_refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_preds, label_ids=formatted_refs)

# === 메인 함수 ===

def main():
    global tokenizer  # 아래 QA 전처리 함수에서 사용하기 위해 전역 변수로 선언
    args = parse_args()

    # dataset_name에 따라 task 및 기본 prefix 결정
    if args.dataset_name in ["samsum", "xsum", "cnn_dailymail"]:
        task = "summarization"
        default_prefix = "summarize: "
    elif args.dataset_name in ["openwebtext", "wikitext-2", "wikitext-103"]:
        task = "text_generation"
        default_prefix = ""
    elif args.dataset_name in ["glue", "superglue"]:
        task = "nlu"
        default_prefix = "classify: "
    elif args.dataset_name == "squad_v1":
        task = "qa"
        default_prefix = "question: "
    elif "wmt" in args.dataset_name:
        if len(args.dataset_name.split('_')) != 3:
            raise ValueError("WMT datasets should be in the format 'wmt19_xx_xx'")
        task = "translation"
        default_prefix = f"translate {args.dataset_name.split('_')[1]} to {args.dataset_name.split('_')[2]}: "
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    args.task = task
    if not hasattr(args, "source_prefix") or args.source_prefix is None:
        args.source_prefix = default_prefix

    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs"
    output_dir = os.path.join("results", exp_name, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1. wandb 초기화
    wandb.init(project=exp_name, name=args.run_name)

    # 2. 데이터셋 로드
    if args.dataset_name == "samsum":
        dataset = load_dataset("samsum", trust_remote_code=True)
    elif args.dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")
    elif args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset = load_dataset("Salesforce/wikitext", name=args.dataset_name+"-raw-v1")
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
    elif "wmt19" in args.dataset_name:
        try:
            dataset = load_dataset("wmt/wmt19", "-".join([args.dataset_name.split('_')[1], args.dataset_name.split('_')[2]]), trust_remote_code=True)
        except:
            dataset = load_dataset("wmt/wmt19", "-".join([args.dataset_name.split('_')[2], args.dataset_name.split('_')[1]]), trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print("Loaded dataset:", dataset)

    # 3. 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")

    # 4. 전처리 함수 설정 (QA 태스크인 경우 별도 전처리 함수 사용)
    if task == "qa":
        # 학습 데이터 전처리
        tokenized_train = dataset["train"].map(
            preprocess_function_qa_train,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=8,
            desc="Tokenizing train dataset for QA"
        )
        # 평가 데이터 전처리 (offset mapping 포함)
        eval_examples = dataset["validation"]
        tokenized_eval = eval_examples.map(
            preprocess_function_qa_eval,
            batched=True,
            remove_columns=eval_examples.column_names,
            num_proc=8,
            desc="Tokenizing eval dataset for QA"
        )
        tokenized_dataset = {"train": tokenized_train, "validation": tokenized_eval}
    else:
        # 그 외 태스크는 기존 전처리 함수 사용
        def preprocess_function(batch):
            if task == "summarization":
                if "dialogue" in batch and "summary" in batch:
                    inputs = tokenizer([args.source_prefix + dialogue for dialogue in batch["dialogue"]], truncation=True)
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(batch["summary"], truncation=True)
                elif "document" in batch and "summary" in batch:
                    inputs = tokenizer([args.source_prefix + doc for doc in batch["document"]], truncation=True)
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(batch["summary"], truncation=True)
                else:
                    inputs = tokenizer([args.source_prefix + article for article in batch["article"]], truncation=True)
                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(batch["highlights"], truncation=True)
                inputs["labels"] = labels["input_ids"]
                return inputs
            elif task == "text_generation":
                if args.source_prefix:
                    inputs = tokenizer([args.source_prefix + text for text in batch["text"]], truncation=True)
                else:
                    inputs = tokenizer(batch["text"], truncation=True)
                inputs["labels"] = inputs["input_ids"].copy()
                return inputs
            elif task == "nlu":
                if "sentence1" in batch and "sentence2" in batch:
                    inputs = tokenizer([args.source_prefix + s for s in batch["sentence1"]], batch["sentence2"], truncation=True)
                elif "premise" in batch and "hypothesis" in batch:
                    inputs = tokenizer([args.source_prefix + p for p in batch["premise"]], batch["hypothesis"], truncation=True)
                elif "sentence" in batch:
                    inputs = tokenizer([args.source_prefix + s for s in batch["sentence"]], truncation=True)
                elif "text" in batch:
                    inputs = tokenizer([args.source_prefix + t for t in batch["text"]], truncation=True)
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
            elif task == "translation":
                src_lang = args.dataset_name.split('_')[1]
                tgt_lang = args.dataset_name.split('_')[2]
                inputs = [args.source_prefix + t[src_lang] for t in batch["translation"]]
                targets = [t[tgt_lang] for t in batch["translation"]]
                model_inputs = tokenizer(inputs, truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(targets, truncation=True)
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=8,
            desc="Tokenizing dataset"
        )
        if "test" not in tokenized_dataset.keys():
            tokenized_dataset["test"] = tokenized_dataset["validation"]

    # 5. Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 6. Training Arguments 설정
    eval_steps = len(tokenized_dataset["train"]) // (args.per_device_train_batch_size * 2)
    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        fp16=args.fp16,
        save_total_limit=3,
    )

    generation_kwargs = {
        "min_length": args.gen_min_length,
        "max_length": args.gen_max_length,
        "no_repeat_ngram_size": args.gen_no_repeat_ngram_size,
        "num_beams": args.gen_num_beams,
    }

    # 7. Trainer 설정 (QA 태스크인 경우 post_process_function 및 eval_examples 전달)
    if task == "qa":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            eval_examples=dataset["validation"],  # 원본 예제 (후처리용)
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if task != "qa" else None,
            post_process_function=post_process_function_qa,
        )
        test_dataset = tokenized_dataset["test"]
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        test_dataset = tokenized_dataset["test"]

    test_callback = TestEvaluationCallback(test_dataset, compute_metrics, tokenizer, task, generation_kwargs, output_dir)
    test_callback.trainer = trainer
    trainer.add_callback(test_callback)

    # 8. 모델 학습 및 평가
    trainer.train()
    trainer.save_model(output_dir)
    
    # 테스트셋 평가 및 예측 (Generation 인자 적용)
    test_results = trainer.predict(test_dataset, **generation_kwargs)
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

    decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
    
    if task == "text_generation":
        final_metrics = {}
    else:
        final_metrics = compute_metrics((predictions, labels))
    
    sample_list = []
    for pred, gold in zip(decoded_preds, decoded_labels):
        sample_list.append({"prediction": pred, "gold": gold})
    samples_file = os.path.join(output_dir, "pred_gold_samples.json")
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(sample_list, f, indent=4, ensure_ascii=False)
    
    results_file = os.path.join(output_dir, f"{task}_switch_results.json")
    with open(results_file, "w") as f:
        json.dump({k: round(v, 4) for k, v in final_metrics.items()}, f, indent=4)
    
    print("Model and results saved.")

if __name__ == "__main__":
    main()
