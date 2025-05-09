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
    T5ForConditionalGeneration,
)
from base_Switch_Transformer import SwitchTransformersForConditionalGeneration, SwitchTransformersConfig
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
        # 5 에폭마다 평가 수행
        if int(state.epoch) % 5 != 0:
            return control

        # if self.trainer is None:
        #     print("Trainer is not set in callback.")
        #     return control

        # predict 호출 시 generation 인자 전달
        test_results = self.trainer.predict(self.test_dataset, **self.generation_kwargs)
        predictions = test_results.predictions
        labels = test_results.label_ids

        if self.task in ["summarization", "qa", "nlu", "translation"]:
            preds = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # postprocessing: 문장 단위 줄바꿈 적용
            decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
        elif self.task == "text_generation":
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        if self.task == "text_generation":
            eval_loss = test_results.metrics.get("eval_loss")
            perplexity = math.exp(eval_loss) if eval_loss is not None else None
            test_metrics = {"perplexity": perplexity}
        else:
            test_metrics = self.compute_metrics((predictions, labels))
        
        sample_list = []
        for pred, gold in zip(decoded_preds, decoded_labels):
            sample_list.append({"prediction": pred, "gold": gold})
        with open(os.path.join(self.output_dir, f"pred_gold_samples_epoch{state.epoch}.json"), "w", encoding="utf-8") as f:
            json.dump(sample_list, f, indent=4, ensure_ascii=False)
            
        results_file = os.path.join(self.output_dir, f"{state.epoch}_switch_results.json")
        with open(results_file, "w") as f:
            json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=4)
        print("Model and results saved.")
            
        print(f"Test metrics at epoch {state.epoch}: {test_metrics}")
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        
        #save model
        self.trainer.save_model(os.path.join(self.output_dir, f"epoch_{state.epoch}"))
        
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
    # Generation 인자 추가
    parser.add_argument("--gen_min_length", type=int, default=10, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    # source_prefix 인자 추가 (summarization 전처리 시 사용)
    parser.add_argument("--source_prefix", type=str, default=None, help="Source prefix to prepend to input text for summarization")
    
    parser.add_argument("--mode", type=str, default="base", choices=["base", "dense", "share"], help="Switch Transformer mode")
    return parser.parse_args()

def main():
    args = parse_args()

    # data# dataset_name에 따라 task와 기본 prefix를 자동으로 결정
    if args.dataset_name in ["samsum", "xsum", "cnn_dailymail"]:
        task = "summarization"
        default_prefix = "summarize: "
    elif args.dataset_name in ["openwebtext", "wikitext-2", "wikitext-103"]:
        task = "text_generation"  # 텍스트 생성(언어모델링) 태스크
        default_prefix = ""         # prefix가 필요없으면 빈 문자열로 설정하거나 "generate: " 등으로 지정 가능
    elif args.dataset_name in ["glue", "superglue"]:
        task = "nlu"
        default_prefix = "classify: "  # 예시: 문장 분류 태스크로 인식하도록 prefix 부여
    elif args.dataset_name == "squad_v1":
        task = "qa"
        default_prefix = "question: "  # 질문에 대한 prefix 부여
    elif "wmt" in args.dataset_name:
        if len(args.dataset_name.split('_')) != 3:
            raise ValueError("WMT datasets should be in the format 'wmt19_xx_xx'")
        task = "translation"
        default_prefix = f"translate {args.dataset_name.split('_')[1]} to {args.dataset_name.split('_')[2]}: "  # 번역에 대한 prefix 부여
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    args.task = task
    # 사용자가 별도로 source_prefix를 지정하지 않았다면 기본값을 사용
    if not hasattr(args, "source_prefix") or args.source_prefix is None:
        args.source_prefix = default_prefix

    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs_final"
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
            dataset = load_dataset("wmt/wmt19", "-".join([args.dataset_name.split('_')[1],args.dataset_name.split('_')[2]]), trust_remote_code=True)
        except:
            dataset = load_dataset("wmt/wmt19", "-".join([args.dataset_name.split('_')[2],args.dataset_name.split('_')[1]]), trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print("Loaded dataset:", dataset)

    # ------------------------------
    # 3. 모델 및 토크나이저 로드 (baseline Switch Transformer)
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.mode == "base":
        model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
    if args.mode == "dense":
        model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
        model.to_dense()
    elif args.mode == "t5":
        if "base" in args.model_name:
            model = T5ForConditionalGeneration.from_pretrained("t5-base", device_map="auto")
        elif "large" in args.model_name:
            model = T5ForConditionalGeneration.from_pretrained("t5-large", device_map="auto")
        elif "xxl" in args.model_name:
            model = T5ForConditionalGeneration.from_pretrained("t5-xxl", device_map="auto")
        else:
            raise ValueError("Dense expert model not found.")
    elif args.mode == "share":
        model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")
        model.make_share_expert()
    print(model)
    
    # ---------------------------------------------------------
    # 전처리 함수 및 평가 지표 (태스크별)
    # ---------------------------------------------------------
    if task == "summarization":
        # 입력 텍스트 앞에 source_prefix 적용 (예: "summarize: ")
        def preprocess_function(batch):
            # 예: samsum, xsum, cnn_dailymail 등
            if "dialogue" in batch and "summary" in batch:
                inputs = tokenizer([args.source_prefix + dialogue for dialogue in batch["dialogue"]], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["summary"], truncation=True)
            elif "document" in batch and "summary" in batch:
                inputs = tokenizer([args.source_prefix + doc for doc in batch["document"]], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["summary"], truncation=True)
            else:
                # 예: XSum는 "article"과 "highlights" 사용
                inputs = tokenizer([args.source_prefix + article for article in batch["article"]], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["highlights"], truncation=True)
            inputs["labels"] = labels["input_ids"]
            return inputs

        rouge_metric = evaluate.load("rouge")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # postprocessing: 문장 단위 줄바꿈 적용
            decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
            result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {key: round(value * 100, 4) for key, value in result.items()}
            return result

    elif task == "text_generation":
        def preprocess_function(batch):
            # 텍스트 생성 태스크의 경우, prefix가 있으면 입력 앞에 추가
            if args.source_prefix:
                inputs = tokenizer([args.source_prefix + text for text in batch["text"]], truncation=True)
            else:
                inputs = tokenizer(batch["text"], truncation=True)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
        def compute_metrics(eval_preds):
            return {}

    elif task == "nlu":
        def preprocess_function(batch):
            # NLU 태스크의 경우, 예시로 sentence1에 prefix를 붙임 (필요에 따라 조정 가능)
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
        # 전처리 함수: 질문과 문맥을 결합하여 T5의 입력 형식으로 변환하고, 첫 번째 정답을 타깃으로 사용합니다.
        def preprocess_function(examples):
            # T5는 text-to-text 모델이므로, 입력 텍스트에 질문과 문맥을 함께 넣어줍니다.
            inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
            # print(inputs)
            # SQuAD의 answers는 리스트 형태이므로 첫 번째 정답을 선택합니다.
            targets = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]
            # print(targets)
            
            # 입력과 타깃 토크나이징 (max_length, truncation 등 필요에 따라 조정)
            model_inputs = tokenizer(inputs, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, truncation=True)
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs


        qa_metric = evaluate.load("squad")
        # def compute_metrics(eval_preds):
        #     preds, labels = eval_preds
        #     if isinstance(preds, tuple):
        #         preds = preds[0]
        #     preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #     result = qa_metric.compute(predictions=decoded_preds, references=decoded_labels)
        #     return result
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # SQuAD 평가 메트릭이 요구하는 형식으로 변환
            formatted_preds = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(decoded_preds)]
            # 실제 데이터에서는 answer_start를 제공해야 하지만, 여기서는 없으므로 0으로 처리합니다.
            formatted_refs = [{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}} for i, ref in enumerate(decoded_labels)]
            
            result = qa_metric.compute(predictions=formatted_preds, references=formatted_refs)
            return result

    elif task == "translation":
        def preprocess_function(batch):
            src_lang = args.dataset_name.split('_')[1]
            tgt_lang = args.dataset_name.split('_')[2]
            inputs = [args.source_prefix + t[src_lang] for t in batch["translation"]]
            targets = [t[tgt_lang] for t in batch["translation"]]
            model_inputs = tokenizer(inputs, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        translation_metric = evaluate.load("sacrebleu")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # sacreBLEU은 각 참조 문장을 리스트로 감싸야 합니다.
            decoded_labels = [[label] for label in decoded_labels]
            result = translation_metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": round(result["score"], 4)}
            return result
    else:
        raise ValueError(f"Unsupported task: {task}")

    # remove_columns는 train split의 컬럼 이름 사용
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names, num_proc=8)
    
    if "test" not in tokenized_dataset.keys():
        tokenized_dataset["test"] = tokenized_dataset["validation"]
    
    # ------------------------------
    # 5. Data Collator
    # ------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


    # ------------------------------
    # 7. Training Arguments 설정
    # ------------------------------
    # eval_steps를 1 epoch당 두 번 평가하도록 동적으로 계산 (train split 길이에 따라)
    eval_steps = len(tokenized_dataset["train"]) // (args.per_device_train_batch_size)
    
    # 태스크별 best model selection 기준 설정
    if task == "summarization":
        metric_for_best_model = "rouge2"
        greater_is_better = True
    elif task == "nlu":
        metric_for_best_model = "accuracy"
        greater_is_better = True
    elif task == "qa":
        metric_for_best_model = "f1"
        greater_is_better = True
    elif task == "translation":
        metric_for_best_model = "bleu"
        greater_is_better = True
    elif task == "text_generation":
        # text_generation의 경우 compute_metrics가 빈 dict를 반환하므로 eval_loss를 사용합니다.
        metric_for_best_model = "eval_loss"
        greater_is_better = False
        
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
        save_steps=eval_steps,
        predict_with_generate=True,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        fp16=args.fp16,
        save_total_limit=1,
        save_safetensors=False,
        load_best_model_at_end=True,                # best model 자동 로드
        metric_for_best_model=metric_for_best_model,  # 평가 기준 metric
        greater_is_better=greater_is_better           # 평가 기준에 따른 우수 모델 결정
    )

    # Generation 인자 딕셔너리 생성
    generation_kwargs = {
        "min_length": args.gen_min_length,
        "max_length": args.gen_max_length,
        "no_repeat_ngram_size": args.gen_no_repeat_ngram_size,
        "num_beams": args.gen_num_beams,
    }

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
    
    test_callback = TestEvaluationCallback(tokenized_dataset["test"], compute_metrics, tokenizer, task, generation_kwargs, output_dir)
    test_callback.trainer = trainer  # trainer 인스턴스를 직접 할당
    trainer.add_callback(test_callback)


    # ------------------------------
    # 9. 모델 학습 및 평가
    # ------------------------------
    trainer.train()
    trainer.save_model(output_dir)
    
    # best checkpoint의 경로에서 global step 추출 (예: "checkpoint-1000")
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint is not None:
        global_step_str = best_checkpoint.split("-")[-1]
        best_global_step = int(global_step_str)
        # 한 에폭 당 update step 수 계산 (batch size에 따른 step 수)
        steps_per_epoch = len(tokenized_dataset["train"]) // args.per_device_train_batch_size
        best_epoch = best_global_step / steps_per_epoch
        print(f"Best model is from approximately epoch {best_epoch:.2f}")
    else:
        print("Best checkpoint 정보가 없습니다.")

    # 테스트셋 평가 및 예측/정답 디코딩 (Generation 인자 적용)
    test_results = trainer.predict(tokenized_dataset["test"], **generation_kwargs)
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

    # Postprocessing 후 평가 (예: 문장 단위 줄바꿈 적용)
    decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
    
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
    
    print(f"Test metrics at epoch of Best Model: {final_metrics}")
    wandb.log({f"best_test_{k}": v for k, v in final_metrics.items()})

if __name__ == "__main__":
    main()
