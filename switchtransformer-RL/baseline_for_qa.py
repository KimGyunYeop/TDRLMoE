import argparse
import numpy as np
import json
import wandb
import evaluate
import nltk
import datasets
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

from trainer_seq2seq_qa import QuestionAnsweringSeq2SeqTrainer

from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
# nltk 문장 토크나이저가 없으면 다운로드
try:
    nltk.download('punkt_tab')
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

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
        test_metrics = test_results.metrics
        
        results_file = os.path.join(self.output_dir, f"{state.epoch}_switch_results.json")
        with open(results_file, "w") as f:
            json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=4)
        print("Model and results saved.")
            
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
    parser.add_argument("--gen_min_length", type=int, default=1, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=64, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    # source_prefix 인자 추가 (summarization 전처리 시 사용)
    parser.add_argument("--source_prefix", type=str, default=None, help="Source prefix to prepend to input text for summarization")
    
    parser.add_argument("--mode", type=str, default="base", choices=["base", "dense", "share"], help="Switch Transformer mode")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset_name == "squad_v1":
        task = "qa"
        default_prefix = "question: "  # 질문에 대한 prefix 부여
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
    if args.dataset_name == "squad_v1":
        dataset = load_dataset("squad")
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
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    # ---------------------------------------------------------
    # 전처리 함수 및 평가 지표 (태스크별)
    # ---------------------------------------------------------
    if task == "qa":
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

        def preprocess_validation_function(examples):
            # T5는 text-to-text 모델이므로, 입력 텍스트에 질문과 문맥을 함께 넣어줍니다.
            inputs = ["question: " + q + " context: " + c for q, c in zip(examples["question"], examples["context"])]
            # print(inputs)
            # SQuAD의 answers는 리스트 형태이므로 첫 번째 정답을 선택합니다.
            targets = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]
            # print(targets)
            
            # 입력과 타깃 토크나이징 (max_length, truncation 등 필요에 따라 조정)
            model_inputs = tokenizer(inputs, truncation=True, return_overflowing_tokens=True, return_offsets_mapping=True)
            labels = tokenizer(targets, truncation=True)
            
            sample_mapping = model_inputs.pop("overflow_to_sample_mapping")
            model_inputs["example_id"] = []
            labels_out = []
            
            for i in range(len(model_inputs["input_ids"])):
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                model_inputs["example_id"].append(examples["id"][sample_index])
                labels_out.append(labels["input_ids"][sample_index])

            model_inputs["labels"] = labels_out
            return model_inputs

        
        if args.dataset_name == "squad_v1":
            qa_metric = evaluate.load("squad")
        elif args.dataset_name == "squad_v2":
            qa_metric = evaluate.load("squad_v2")
            
        def compute_metrics(p: EvalPrediction):
            return qa_metric.compute(predictions=p.predictions, references=p.label_ids)
        
    else:
        raise ValueError(f"Unsupported task: {task}")

    # remove_columns는 train split의 컬럼 이름 사용
    tokenized_dataset = {}
    tokenized_dataset["train"] = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names, num_proc=8)
    tokenized_dataset["validation"] = dataset["validation"].map(preprocess_validation_function, batched=True, remove_columns=dataset["validation"].column_names, num_proc=8)
    
    if "test" in tokenized_dataset.keys():
        tokenized_dataset["test"] = dataset["test"].map(preprocess_validation_function, batched=True, remove_columns=dataset["test"].column_names, num_proc=8)
    else:
        tokenized_dataset["test"] = tokenized_dataset["validation"]
    
    # Post-processing:
    def post_processing_function(
        examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        # Format the result to the format the metric expects.
        if args.dataset_name == "squad_v2":
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

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
    trainer = QuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        post_process_function=post_processing_function,
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
    final_metrics = test_results.matrics
    
    # 결과 저장
    results_file = os.path.join(output_dir, f"{task}_switch_results.json")
    with open(results_file, "w") as f:
        json.dump({k: round(v, 4) for k, v in final_metrics.items()}, f, indent=4)
    print("Model and results saved.")
    
    print(f"Test metrics at epoch of Best Model: {final_metrics}")
    wandb.log({f"best_test_{k}": v for k, v in final_metrics.items()})

if __name__ == "__main__":
    main()
