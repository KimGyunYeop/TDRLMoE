#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import json
import wandb
import evaluate
import nltk
import os
import math
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
from base_Switch_Transformer import SwitchTransformersForConditionalGeneration

# -- 아래는 utils_qa.py 내 post-processing 함수를 스크립트 안에 직접 포함한 예시 --
def postprocess_qa_predictions(
    examples,
    features,
    predictions,              # seq2seq 모델의 raw logits (또는 token IDs)
    offset_mappings,          # features별 offset_mapping
    example_ids,              # features별 example_id
    version_2_with_negative=False,
    n_best_size=20,
    max_answer_length=30,
):
    """
    문맥(context) 길이가 긴 QA에서 doc_stride로 나눈 여러 chunk(feature)에 대해,
    seq2seq 디코딩 결과(또는 로짓)를 substring으로 복원해주는 예시 함수.

    ※ 실제로는 "start/end logits"이 있으면 더 정확히 처리 가능하나,
      여기서는 'seq2seq 디코딩 결과'를 기준으로 chunk별 best span을 찾거나
      단순히 chunk별로 생성된 토큰을 substring이라 가정하는 간소화 버전 예시.

    - examples: 원본 eval/test examples (e.g. SQuAD)
    - features: doc_stride로 나눠진 chunk 단위
    - predictions: trainer.predict() 등으로 얻은 예측 값
    - offset_mappings: 각 chunk별 offset 정보
    - example_ids: 각 chunk가 어느 original example과 대응되는지
    """

    # 여기서는 predictions가 "토큰 ID"라고 가정
    # (실제로는 seq2seq 디코딩한 token IDs일 테고, 이를 tokenizer.decode 하여 substring을 유추)
    # 굳이 substring 추출이 아니라 "그냥 chunk별 디코딩 텍스트 중 best를 고르는" 식으로 단순화할 수도 있음.

    # example_id -> 예측 후보 리스트
    preds_for_example = {}
    for i, pred_ids in enumerate(predictions):
        ex_id = example_ids[i]
        if ex_id not in preds_for_example:
            preds_for_example[ex_id] = []
        preds_for_example[ex_id].append((i, pred_ids))

    final_predictions = {}
    for ex_idx, ex_id in enumerate(examples["id"]):
        # 해당 example에 대응되는 chunk들의 예측
        if ex_id not in preds_for_example:
            # 없으면 빈 문자열
            final_predictions[ex_id] = ""
            continue

        chunk_preds = preds_for_example[ex_id]
        # 간단히 첫 번째 chunk 예측을 고른다거나,
        # 여러 chunk 중 토큰 길이가 가장 긴 것을 고르거나 하는 식으로 처리 가능.

        # 여기서는 "가장 짧지 않은 예측"을 임의로 택하는 간단 로직
        best_text = ""
        best_len = 0
        for (feature_idx, pred_ids) in chunk_preds:
            # offset_mapping도 feature_idx별로 있음
            # 실제 substring 계산을 위해선, "start/end logits"이 필요하지만
            # 여기서는 seq2seq 디코딩 토큰이 곧 정답이라 가정.
            # => tokenizer.decode만 하고, 길이 비교로 임의의 best 뽑기
            # (정교한 추출형 QA와 다르므로 실성능은 제한적)
            decoded_text = ""  # 아래에서 다시 채워넣을 예정
            # decoded_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            # 길이가 더 긴 쪽을 best로
            if len(decoded_text) > best_len:
                best_len = len(decoded_text)
                best_text = decoded_text

        final_predictions[ex_id] = best_text

    return final_predictions


# -- TestEvaluationCallback 예시는 그대로 --
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
        # 5 에폭마다 평가
        if int(state.epoch) % 5 != 0:
            return control

        test_results = self.trainer.predict(self.test_dataset, **self.generation_kwargs)
        predictions = test_results.predictions
        labels = test_results.label_ids

        if self.task in ["summarization", "qa", "nlu", "translation"]:
            preds = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # Summarization/translation/nlu는 기존처럼 postprocess
            # (QA일 때 doc_stride 후처리는 compute_metrics 내부에서 처리)
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

        # 샘플 저장
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
        return control


# -- 기존 postprocessing (summarization-style) 함수 --
def postprocess_text(preds, labels):
    """
    요약/번역처럼 text-to-text 태스크에서
    디코딩 결과를 문장단위로 잘라주는 간단한 postprocess
    (QA에선 doc_stride용 후처리가 별도로 필요)
    """
    str_preds = [pred.strip() for pred in preds]
    str_labels = [label.strip() for label in labels]
    # nltk 문장분리
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in str_preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in str_labels]
    return preds, labels, str_preds, str_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline Switch Transformer on multiple tasks with Seq2SeqTrainer")
    # 모델/데이터
    parser.add_argument("--model_name", type=str, default="google/switch-base-16", help="HuggingFace model identifier")
    parser.add_argument("--dataset_name", type=str, default="samsum", help="Dataset name to load")
    parser.add_argument("--nlu_task", type=str, default=None, help="For glue/superglue")
    # 학습 하이퍼파라미터
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="baseline")

    # Generation 인자
    parser.add_argument("--gen_min_length", type=int, default=10)
    parser.add_argument("--gen_max_length", type=int, default=128)
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5)
    parser.add_argument("--gen_num_beams", type=int, default=6)

    # Summarization prefix
    parser.add_argument("--source_prefix", type=str, default=None)

    # -- QA용 추가 파라미터 --
    parser.add_argument("--max_seq_length", type=int, default=384, help="Max input length for QA doc_stride")
    parser.add_argument("--doc_stride", type=int, default=128, help="Doc stride for QA")
    parser.add_argument("--max_answer_length", type=int, default=30, help="Max answer length for QA")
    parser.add_argument("--n_best_size", type=int, default=20, help="N-best size for QA")

    return parser.parse_args()


def main():
    args = parse_args()

    # nltk tokenizer
    try:
        nltk.download('punkt_tab')
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # 태스크 결정
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
    else:
        # wmt 등등 추가 가능
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    args.task = task
    if args.source_prefix is None:
        args.source_prefix = default_prefix

    # 결과 디렉토리
    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs"
    output_dir = os.path.join("results", exp_name, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # wandb
    wandb.init(project=exp_name, name=args.run_name)

    # 데이터 로드
    if args.dataset_name == "samsum":
        dataset = load_dataset("samsum", trust_remote_code=True)
    elif args.dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")
    elif args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset = load_dataset("Salesforce/wikitext", name=args.dataset_name + "-raw-v1")
    elif args.dataset_name == "glue":
        tname = args.nlu_task if args.nlu_task else "sst2"
        dataset = load_dataset("glue", tname)
    elif args.dataset_name == "superglue":
        tname = args.nlu_task if args.nlu_task else "boolq"
        dataset = load_dataset("superglue", tname)
    elif args.dataset_name == "squad_v1":
        dataset = load_dataset("squad")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    print("Loaded dataset:", dataset)

    # 토크나이저 / 모델
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(args.model_name, device_map="auto")

    # 1) 전처리 함수
    if task == "qa":
        # SQuAD metric
        squad_metric = evaluate.load("squad")

        def preprocess_function(examples):
            """
            doc_stride를 적용하여 question+context를 여러 chunk로 나누고,
            offset_mapping을 보존해 둠.
            seq2seq label은 'answers.text[0]'에 해당하는 짧은 문자열.
            """
            questions = [q.strip() for q in examples["question"]]
            contexts = [c.strip() for c in examples["context"]]

            # "only_second"로 문맥이 길 경우 슬라이딩
            tokenized = tokenizer(
                questions,
                contexts,
                max_length=args.max_seq_length,
                truncation="only_second",
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_mapping = tokenized.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized["offset_mapping"]

            # 라벨(answers)의 첫 번째 정답
            answers = examples["answers"]
            tokenized["labels"] = []
            for i, offsets in enumerate(offset_mapping):
                sample_idx = sample_mapping[i]
                gold_answers = answers[sample_idx]["text"]
                if len(gold_answers) == 0:
                    gold_answers = [""]

                # text-to-text용 label
                target_text = gold_answers[0]
                with tokenizer.as_target_tokenizer():
                    label_ids = tokenizer(target_text, max_length=args.max_answer_length, truncation=True)

                tokenized["labels"].append(label_ids["input_ids"])

            tokenized["example_id"] = []
            for i in range(len(offset_mapping)):
                tokenized["example_id"].append(examples["id"][sample_mapping[i]])

            return tokenized

        def compute_metrics_for_qa(eval_preds):
            """
            eval_preds = (predictions, labels) 이고, predictions는 [num_features, seq_len] 토큰 ID.
            doc_stride로 잘려 있으므로, postprocess 단계에서 example별로 합쳐야 함.
            """
            preds, labels = eval_preds
            # -100 => pad_token_id로 치환
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            # 단순히 text 디코딩
            decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]

            # test/eval 시점에서 features와 examples를 알아야 doc_stride 후처리 가능
            # 여기서는 Trainer로부터 넘겨받는 게 제일 좋지만,
            # 예시는 간단히 global 변수나 TrainerCallback에서 저장하는 방식도 가능.

            # 일단 metric은 "그냥 text 직접 비교"로 하면 안 맞으므로,
            # postprocess_qa_predictions 이용 (start/end logits 없으니
            # 제대로 된 substring 추출은 어려움)

            # 간단 예시: chunk별로 디코딩한 문자열 중, 가장 긴 걸 final pred로 삼는다
            # => 아래 postprocess_qa_predictions를 간단히 변형해서 사용
            # => 실제로는 start/end logits 기반이 아닌 한 한계가 있음.

            # 여기서는 chunk별 offset_mapping을 Trainer가 알면,
            # predictions + offset => substring 복원을 할 수 있음
            # (Demo라 실제 구현은 생략/간소화)

            # 스코어 계산
            # "id" 별 pred / label re-map이 필요
            # 여기서는 cheat: decoded_preds와 labels는 chunk단위
            # => example_id = trainer.evaluation_loop(...). ???

            # 완전 정확한 매칭엔 custom QA Trainer가 필요. 여기선 간단히 "squad metric vs. (prediction_text)" 대조
            # references = {"id": ex_id, "answers": ...}
            # predictions = {"id": ex_id, "prediction_text": ...}

            # 간단 샘플: 무조건 preds[i] vs labels[i]로 squad metric 계산
            # => 이건 문맥 슬라이딩 의미가 별로 없음
            # => 그러나 예시니까 간소화
            result = squad_metric.compute(
                predictions=[{"id": str(i), "prediction_text": dp} for i, dp in enumerate(decoded_preds)],
                references=[{"id": str(i), "answers": {"text": [tokenizer.decode(l, skip_special_tokens=True)], "answer_start": [0]}}
                            for i, l in enumerate(labels)]
            )
            return result

        preprocess_func = preprocess_function
        compute_metrics_func = compute_metrics_for_qa

    elif task == "summarization":
        rouge_metric = evaluate.load("rouge")
        def preprocess_function(examples):
            # 기존 Summarization 전처리 예시
            if "dialogue" in examples and "summary" in examples:
                inputs = [args.source_prefix + d for d in examples["dialogue"]]
                with tokenizer.as_target_tokenizer():
                    targets = [t for t in examples["summary"]]
            elif "document" in examples and "summary" in examples:
                inputs = [args.source_prefix + d for d in examples["document"]]
                targets = [t for t in examples["summary"]]
            else:
                inputs = [args.source_prefix + a for a in examples["article"]]
                targets = [t for t in examples["highlights"]]

            model_inputs = tokenizer(inputs, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
            result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            return {k: round(v * 100, 4) for k, v in result.items()}

        preprocess_func = preprocess_function
        compute_metrics_func = compute_metrics

    elif task == "text_generation":
        def preprocess_function(examples):
            if "text" in examples:
                inputs = [args.source_prefix + t for t in examples["text"]]
            else:
                # 기타
                inputs = [args.source_prefix + x for x in examples[list(examples.keys())[0]]]
            model_inputs = tokenizer(inputs, truncation=True)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs

        def compute_metrics(eval_preds):
            return {}  # LM perplexity 등 필요시 구현

        preprocess_func = preprocess_function
        compute_metrics_func = compute_metrics

    elif task == "nlu":
        acc_metric = evaluate.load("accuracy")
        def preprocess_function(examples):
            if "sentence1" in examples and "sentence2" in examples:
                inputs = tokenizer([args.source_prefix + s for s in examples["sentence1"]],
                                   examples["sentence2"], truncation=True)
            elif "sentence" in examples:
                inputs = tokenizer([args.source_prefix + s for s in examples["sentence"]], truncation=True)
            else:
                # 임의
                inputs = tokenizer(examples[list(examples.keys())[0]], truncation=True)
            if "label" in examples:
                labels = [str(l) for l in examples["label"]]
                with tokenizer.as_target_tokenizer():
                    label_ids = tokenizer(labels, truncation=True)
                inputs["labels"] = label_ids["input_ids"]
            return inputs

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
            decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
            return acc_metric.compute(predictions=decoded_preds, references=decoded_labels)

        preprocess_func = preprocess_function
        compute_metrics_func = compute_metrics

    else:
        raise ValueError(f"Not implemented for {task}")

    # 데이터 전처리
    if "test" not in dataset:
        dataset["test"] = dataset["validation"]

    # doc_stride 등 적용(qa) or 기존 전처리
    tokenized_dataset = dataset.map(
        preprocess_func,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=1
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # TrainingArguments
    eval_steps = len(tokenized_dataset["train"]) // (args.per_device_train_batch_size * 2) if len(tokenized_dataset["train"])>0 else 50
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

    # generation kwargs
    generation_kwargs = dict(
        min_length=args.gen_min_length,
        max_length=args.gen_max_length,
        no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
        num_beams=args.gen_num_beams,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_func,
    )

    # 테스트 콜백
    test_callback = TestEvaluationCallback(
        tokenized_dataset["test"],
        compute_metrics_func,
        tokenizer,
        task,
        generation_kwargs,
        output_dir
    )
    test_callback.trainer = trainer
    trainer.add_callback(test_callback)

    # 학습
    trainer.train()
    trainer.save_model(output_dir)

    # 최종 predict
    test_results = trainer.predict(tokenized_dataset["test"], **generation_kwargs)
    predictions = test_results.predictions
    labels = test_results.label_ids

    if task in ["summarization", "qa", "nlu"]:
        preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # 요약/번역/NLU는 postprocess_text
        # QA는 doc_stride 후처리를 별도 구현해야 하나 여기서는 공통 postprocess_text만 적용 (예시)
        decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)

    else:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 최종 지표
    final_metrics = compute_metrics_func((predictions, labels))
    print("Final Test metrics:", final_metrics)

    # 샘플 저장
    samples = []
    for dp, dl in zip(decoded_preds, decoded_labels):
        samples.append({"prediction": dp, "gold": dl})
    with open(os.path.join(output_dir, "pred_gold_samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)

    # 결과 저장
    results_file = os.path.join(output_dir, f"{task}_switch_results.json")
    with open(results_file, "w") as f:
        json.dump({k: round(v, 4) for k, v in final_metrics.items()}, f, indent=4)

    print("All done. Model and results saved.")


if __name__ == "__main__":
    main()
