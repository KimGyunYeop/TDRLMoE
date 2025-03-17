import argparse
import os
import json
import numpy as np
import nltk
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from base_Switch_Transformer import SwitchTransformersForConditionalGeneration

# nltk 문장 토크나이저 확인
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def postprocess_text(preds, labels):
    """
    예측문과 정답 문장을 각각 strip한 후, nltk.sent_tokenize를 사용해 문장 단위로 분리하고,
    각 문장 사이에 줄바꿈("\n")을 추가합니다.
    """
    str_preds = [pred.strip() for pred in preds]
    str_labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in str_preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in str_labels]
    return preds, labels, str_preds, str_labels

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a saved checkpoint and test with hyper_MoE-like preprocessing & generation settings."
    )
    # 필수 인자
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory of the saved checkpoint (contains tokenizer & model).")

    # 데이터셋 정보
    parser.add_argument("--dataset_name", type=str, default="cnn_dailymail",
                        help="Dataset name to test (e.g., samsum, xsum, cnn_dailymail).")
    parser.add_argument("--dataset_config", type=str, default="3.0.0",
                        help="Dataset config name (e.g. '3.0.0' for cnn_dailymail).")
    parser.add_argument("--task", type=str, choices=["summarization", "text_generation", "nlu", "qa"],
                        default="summarization", help="Task type.")
    parser.add_argument("--nlu_task", type=str, default=None,
                        help="For glue/superglue, specify subtask name (e.g., sst2, boolq).")

    # 전처리 파라미터 (hyper_MoE 스타일)
    parser.add_argument("--max_source_length", type=int, default=1024, help="Max source text length.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max target text length.")
    parser.add_argument("--ignore_pad_token_for_loss", action="store_true", default=False,
                        help="Replace pad_token_id in labels with -100 to ignore them in the loss.")

    # Generation 파라미터 (hyper_MoE 스타일)
    parser.add_argument("--gen_min_length", type=int, default=10, help="Minimum generation length.")
    parser.add_argument("--gen_max_length", type=int, default=60, help="Maximum generation length.")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=3, help="No repeat ngram size.")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation.")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty.")

    # 배치 사이즈
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Evaluation batch size per device.")
    # 테스트셋 샘플 제한
    parser.add_argument("--max_test_samples", type=int, default=320,
                        help="Max number of samples to use from test set (0 or negative means use all).")

    return parser.parse_args()

def main():
    args = parse_args()

    # ------------------------------
    # 1. 데이터셋 로드
    # ------------------------------
    if args.dataset_name == "samsum":
        dataset = load_dataset("samsum")
    elif args.dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")
    elif args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset = load_dataset(args.dataset_name)
    elif args.dataset_name == "glue":
        task_name = args.nlu_task if args.nlu_task else "sst2"
        dataset = load_dataset("glue", task_name)
    elif args.dataset_name == "superglue":
        task_name = args.nlu_task if args.nlu_task else "boolq"
        dataset = load_dataset("superglue", task_name)
    elif args.dataset_name == "squad_v1":
        dataset = load_dataset("squad")
    elif args.dataset_name == "xsum":
        dataset = load_dataset("xsum", trust_remote_code=True)
    elif args.dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", args.dataset_config)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # test split 있으면 test, 없으면 validation, 그것도 없으면 train
    if "test" in dataset:
        test_dataset = dataset["test"]
    elif "validation" in dataset:
        test_dataset = dataset["validation"]
    else:
        test_dataset = dataset["train"]

    # ------------------------------
    # 2. 토크나이저 / 모델 불러오기
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(args.checkpoint_dir)

    # ------------------------------
    # 3. 전처리 함수 정의 (hyper_MoE 스타일)
    #    - summarization인 경우 article/document와 highlights/summary 사용
    #    - tokenizer(..., max_length=..., padding="max_length", truncation=True)
    #    - labels를 -100으로 치환 (ignore_pad_token_for_loss=True 시)
    # ------------------------------
    if args.task == "summarization":
        def preprocess_function(examples):
            # cnn_dailymail일 경우 "article", "highlights"
            # xsum일 경우 "document", "summary"
            # samsum일 경우 "dialogue", "summary"
            # (아래는 기본적으로 cnn/dailymail 가정)
            if "article" in examples and "highlights" in examples:
                inputs = examples["article"]
                targets = examples["highlights"]
            elif "document" in examples and "summary" in examples:
                inputs = examples["document"]
                targets = examples["summary"]
            elif "dialogue" in examples and "summary" in examples:
                inputs = examples["dialogue"]
                targets = examples["summary"]
            else:
                # 데이터셋별로 다른 키를 사용한다면 필요에 맞게 수정
                # (예: xsum: "document"/"summary", samsum: "dialogue"/"summary" 등)
                # 아래는 fallback 로직
                col_names = list(examples.keys())
                if len(col_names) >= 2:
                    inputs = examples[col_names[0]]
                    targets = examples[col_names[1]]
                else:
                    # 예외 처리
                    inputs = examples[col_names[0]]
                    targets = [""] * len(inputs)

            # 인코딩
            model_inputs = tokenizer(
                inputs,
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True
            )
            # 타깃 인코딩
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=args.max_target_length,
                    padding="max_length",
                    truncation=True
                )

            # -100 처리
            if args.ignore_pad_token_for_loss:
                labels_ids = []
                for label_seq in labels["input_ids"]:
                    label_seq = [
                        (l if l != tokenizer.pad_token_id else -100)
                        for l in label_seq
                    ]
                    labels_ids.append(label_seq)
                labels["input_ids"] = labels_ids

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    elif args.task == "text_generation":
        def preprocess_function(examples):
            # 텍스트 필드 (openwebtext: "text", wikitext 등)
            col = "text"
            if col not in examples:
                col = list(examples.keys())[0]
            inputs = tokenizer(
                examples[col],
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True
            )
            if args.ignore_pad_token_for_loss:
                labels_ids = []
                for label_seq in inputs["input_ids"]:
                    label_seq = [
                        (l if l != tokenizer.pad_token_id else -100)
                        for l in label_seq
                    ]
                    labels_ids.append(label_seq)
                inputs["labels"] = labels_ids
            else:
                # 텍스트 생성 태스크에서는 보통 inputs == labels
                inputs["labels"] = inputs["input_ids"].copy()
            return inputs

    elif args.task == "nlu":
        def preprocess_function(examples):
            # 예: GLUE/SuperGLUE 등. 
            # multi-sentence인 경우 tokenizer(sentence1, sentence2)
            # 라벨은 정수 -> 문자열 변환 후 tokenizer
            if "sentence1" in examples and "sentence2" in examples:
                inputs = tokenizer(
                    examples["sentence1"], examples["sentence2"],
                    max_length=args.max_source_length,
                    padding="max_length",
                    truncation=True
                )
            elif "premise" in examples and "hypothesis" in examples:
                inputs = tokenizer(
                    examples["premise"], examples["hypothesis"],
                    max_length=args.max_source_length,
                    padding="max_length",
                    truncation=True
                )
            elif "sentence" in examples:
                inputs = tokenizer(
                    examples["sentence"],
                    max_length=args.max_source_length,
                    padding="max_length",
                    truncation=True
                )
            else:
                col = list(examples.keys())[0]
                inputs = tokenizer(
                    examples[col],
                    max_length=args.max_source_length,
                    padding="max_length",
                    truncation=True
                )
            # 라벨 처리
            if "label" in examples:
                labels_str = [str(l) for l in examples["label"]]
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        labels_str,
                        max_length=args.max_target_length,
                        padding="max_length",
                        truncation=True
                    )
                if args.ignore_pad_token_for_loss:
                    labels_ids = []
                    for label_seq in labels["input_ids"]:
                        label_seq = [
                            (l if l != tokenizer.pad_token_id else -100)
                            for l in label_seq
                        ]
                        labels_ids.append(label_seq)
                    labels["input_ids"] = labels_ids
                inputs["labels"] = labels["input_ids"]
            else:
                inputs["labels"] = None
            return inputs

    elif args.task == "qa":
        def preprocess_function(examples):
            if "question" in examples and "context" in examples:
                inputs = tokenizer(
                    examples["question"], examples["context"],
                    max_length=args.max_source_length,
                    padding="max_length",
                    truncation=True
                )
                answers = [ans[0] if len(ans) > 0 else "" for ans in examples["answers"]["text"]]
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        answers,
                        max_length=args.max_target_length,
                        padding="max_length",
                        truncation=True
                    )
                if args.ignore_pad_token_for_loss:
                    labels_ids = []
                    for label_seq in labels["input_ids"]:
                        label_seq = [
                            (l if l != tokenizer.pad_token_id else -100)
                            for l in label_seq
                        ]
                        labels_ids.append(label_seq)
                    labels["input_ids"] = labels_ids
                inputs["labels"] = labels["input_ids"]
            else:
                # fallback
                inputs = {}
            return inputs

    # ------------------------------
    # 4. 전처리 map + test 샘플 제한
    # ------------------------------
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=test_dataset.column_names
    )
    if args.max_test_samples > 0 and len(test_dataset) > args.max_test_samples:
        test_dataset = test_dataset.select(range(args.max_test_samples))

    # ------------------------------
    # 5. DataCollator
    # ------------------------------
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id
    )

    # ------------------------------
    # 6. Trainer 설정 (predict only)
    # ------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=False,
        do_train=False,
        do_eval=False,
        do_predict=True,
        # 로그/체크포인트 최소화
        logging_steps=1000000,  # 거의 로그 안 뜨도록
        save_steps=1000000,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ------------------------------
    # 7. 테스트셋 추론 (hyper_MoE-style generation 파라미터)
    #    min_length, max_length, no_repeat_ngram_size, num_beams, length_penalty 등
    # ------------------------------
    predict_results = trainer.predict(
        test_dataset,
        max_length=args.gen_max_length,
        min_length=args.gen_min_length,
        no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
        num_beams=args.gen_num_beams,
        length_penalty=args.length_penalty
    )
    predictions = predict_results.predictions
    labels = predict_results.label_ids

    # ------------------------------
    # 8. postprocess & 평가 (summarization이면 ROUGE)
    # ------------------------------
    if args.task in ["summarization", "qa", "nlu"]:
        # -100 => pad_token_id로 복원
        preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
    elif args.task == "text_generation":
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE 평가 (summarization)
    if args.task == "summarization":
        rouge_metric = evaluate.load("rouge")
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        print(result)
        result = {key: round(value * 100, 4) for key, value in result.items()}
        print("[ROUGE Scores]", result)
    else:
        print("Decoded predictions (sample) :", decoded_preds[:5])
        print("Decoded labels (sample)      :", decoded_labels[:5])

    # ------------------------------
    # 9. 결과 저장
    # ------------------------------
    output_dir = os.path.join(args.checkpoint_dir, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(decoded_preds, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(decoded_labels, f, indent=4, ensure_ascii=False)

    print("Test results saved to:", output_dir)

if __name__ == "__main__":
    main()
