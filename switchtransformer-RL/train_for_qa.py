import argparse
import math
import numpy as np
import json
import wandb
import evaluate
import os
import torch
import nltk
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    Seq2SeqTrainer,
)
from Custom_MoE_SwitchT import SwitchTransformersForConditionalGeneration, SwitchTransformersConfig

# nltk 문장 토크나이저가 없으면 다운로드
try:
    nltk.download('punkt_tab')
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def postprocess_text(preds, labels):
    """
    요약/번역 등을 위해 문장 단위로 개행을 넣어주는 후처리.
    (QA에는 별도 doc_stride 처리 필요하나, 여기선 간소화)
    """
    str_preds = [pred.strip() for pred in preds]
    str_labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in str_preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in str_labels]
    return preds, labels, str_preds, str_labels


# ---------------------------------------------------------
# Custom Trainer (추가 loss들을 wandb에 로깅)
# ---------------------------------------------------------
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
            "sample_lm_loss",
        ]:
            val = getattr(outputs, loss_name, None)
            if val is not None:
                log_dict[loss_name] = val.detach().float().mean().item()
        # 일정 스텝마다만 로깅
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(log_dict)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------
# 5 epoch마다 Test셋 평가하는 콜백 (기존 구조)
# ---------------------------------------------------------
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

        test_results = self.trainer.predict(self.test_dataset, **self.generation_kwargs)
        predictions = test_results.predictions
        labels = test_results.label_ids

        if self.task in ["summarization", "qa", "nlu", "translation"]:
            preds = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # Summarization/QA 등에서 문장 단위 줄바꿈
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


# ---------------------------------------------------------
# RL 활성화 시점 조절 콜백
# ---------------------------------------------------------
class RLActivationCallback(TrainerCallback):
    def __init__(self, do_RL, RL_start_epoch):
        self.do_RL = do_RL
        self.RL_start_epoch = RL_start_epoch

    def on_epoch_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None and hasattr(self, "trainer"):
            model = self.trainer.model
        if self.do_RL:
            if state.epoch >= self.RL_start_epoch:
                model.config.do_RL = True
                model.do_RL = True
                print(f"Epoch {state.epoch:.2f}: RL 활성화 (config.do_RL={model.config.do_RL})")
            else:
                model.config.do_RL = False
                model.do_RL = False
                print(f"Epoch {state.epoch:.2f}: RL 비활성화")
        return control


# ---------------------------------------------------------
# 인자 파싱 (QA doc_stride 등 추가)
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Switch Transformer with doc_stride for QA, etc.")
    parser.add_argument("--model_name", type=str, default="google/switch-base-16", help="Model ID")
    parser.add_argument("--dataset_name", type=str, default="samsum", help="Dataset name to load")
    parser.add_argument("--nlu_task", type=str, default=None, help="For glue/superglue sub-task")

    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="run")

    # RL 인자
    parser.add_argument("--do_RL", action="store_true", default=False)
    parser.add_argument("--RL_expert_change_ratio", type=float, default=0.1)
    parser.add_argument("--RL_sample_num", type=int, default=4)
    parser.add_argument("--RL_loss_coef", type=float, default=1.0)
    parser.add_argument("--RL_sample_stretegy", type=str, default="multinomial", choices=["multinomial", "random"])
    parser.add_argument("--RL_base_logit_type", type=str, default="top1", choices=["top1", "mean"])
    parser.add_argument("--RL_reward_stretegy", type=str, default="minus", choices=["minus", "static", "positive", "clamp"])
    parser.add_argument("--use_sample_lm_loss", action="store_true", default=False)
    parser.add_argument("--RL_start_epoch", type=int, default=0)
    parser.add_argument("--RL_algo", default="reinforce", choices=["reinforce", "ppo"])
    parser.add_argument("--RL_ppo_eps", type=float, default=0.2)

    # Generation 파라미터
    parser.add_argument("--gen_min_length", type=int, default=10)
    parser.add_argument("--gen_max_length", type=int, default=128)
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5)
    parser.add_argument("--gen_num_beams", type=int, default=6)

    # Summarization prefix
    parser.add_argument("--source_prefix", type=str, default=None)

    # ✅ QA doc_stride 관련 추가
    parser.add_argument("--max_seq_length", type=int, default=384, help="Max input length for QA doc_stride")
    parser.add_argument("--doc_stride", type=int, default=128, help="Doc stride size for QA")
    parser.add_argument("--max_answer_length", type=int, default=30, help="Max answer length for QA")
    parser.add_argument("--n_best_size", type=int, default=20, help="N-best size for QA")

    return parser.parse_args()


def main():
    args = parse_args()

    # dataset_name => task 결정
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
        # 번역, etc
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    args.task = task
    if args.source_prefix is None:
        args.source_prefix = default_prefix

    # wandb run name
    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs"
    if args.do_RL:
        run_name_parts = [
            "RL", args.RL_sample_stretegy,
            f"exp{args.RL_expert_change_ratio}",
            f"num{args.RL_sample_num}",
            f"coef{args.RL_loss_coef}",
            args.RL_base_logit_type,
            args.RL_reward_stretegy,
            f"startRL{args.RL_start_epoch}",
        ]
        if args.use_sample_lm_loss:
            run_name_parts.append("samplm")
        args.run_name += '_' + '_'.join(run_name_parts)

    output_dir = os.path.join("results", exp_name, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

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

    # 모델 설정
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = SwitchTransformersConfig.from_pretrained(args.model_name)
    model_config.do_RL = args.do_RL
    model_config.RL_expert_change_ratio = args.RL_expert_change_ratio
    model_config.RL_sample_num = args.RL_sample_num
    model_config.RL_loss_coef = args.RL_loss_coef
    model_config.RL_sample_stretegy = args.RL_sample_stretegy
    model_config.RL_base_logit_type = args.RL_base_logit_type
    model_config.RL_reward_stretegy = args.RL_reward_stretegy
    model_config.use_sample_lm_loss = args.use_sample_lm_loss
    model_config.RL_start_epoch = args.RL_start_epoch
    model_config.RL_algo = args.RL_algo
    model_config.RL_ppo_eps = args.RL_ppo_eps
    print("Model config:", model_config)

    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        config=model_config,
        device_map="auto"
    )

    # RL 활성화 초기상태
    if args.do_RL and 0 >= args.RL_start_epoch:
        model.config.do_RL = True
        model.do_RL = True
        print("초기 RL 활성화")
    else:
        model.config.do_RL = False
        model.do_RL = False
        print("초기 RL 비활성화")

    # 전처리 & metric
    if task == "summarization":
        rouge = evaluate.load("rouge")
        def preprocess_function(batch):
            if "dialogue" in batch and "summary" in batch:
                inputs = tokenizer([args.source_prefix + d for d in batch["dialogue"]], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["summary"], truncation=True)
            elif "document" in batch and "summary" in batch:
                inputs = tokenizer([args.source_prefix + d for d in batch["document"]], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["summary"], truncation=True)
            else:
                inputs = tokenizer([args.source_prefix + a for a in batch["article"]], truncation=True)
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(batch["highlights"], truncation=True)
            inputs["labels"] = labels["input_ids"]
            return inputs

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
            result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            return {k: round(v * 100, 4) for k, v in result.items()}

    elif task == "text_generation":
        def preprocess_function(batch):
            # prefix 붙여 text 시퀀스 생성
            if "text" in batch:
                inputs = [args.source_prefix + t for t in batch["text"]]
            else:
                key = list(batch.keys())[0]
                inputs = [args.source_prefix + x for x in batch[key]]
            enc = tokenizer(inputs, truncation=True)
            enc["labels"] = enc["input_ids"].copy()
            return enc

        def compute_metrics(eval_preds):
            return {}

    elif task == "nlu":
        metric_acc = evaluate.load("accuracy")
        def preprocess_function(batch):
            if "sentence1" in batch and "sentence2" in batch:
                enc = tokenizer(
                    [args.source_prefix + s for s in batch["sentence1"]],
                    batch["sentence2"],
                    truncation=True
                )
            elif "sentence" in batch:
                enc = tokenizer([args.source_prefix + s for s in batch["sentence"]], truncation=True)
            else:
                key = list(batch.keys())[0]
                enc = tokenizer(batch[key], truncation=True)

            if "label" in batch:
                with tokenizer.as_target_tokenizer():
                    lbl = tokenizer([str(l) for l in batch["label"]], truncation=True)
                enc["labels"] = lbl["input_ids"]
            return enc

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = [p.strip() for p in decoded_preds]
            decoded_labels = [l.strip() for l in decoded_labels]
            result = metric_acc.compute(predictions=decoded_preds, references=decoded_labels)
            return result

    elif task == "qa":
        # ✅ SQuAD + doc_stride 예시
        squad_metric = evaluate.load("squad")

        def preprocess_function(examples):
            """
            doc_stride를 사용해 context를 여러 chunk로 나눈 뒤,
            QA 모델 입력 "question: ... context: ..." 형식으로 만듭니다.
            """
            questions = [q.strip() for q in examples["question"]]
            contexts = [c.strip() for c in examples["context"]]

            # doc_stride 적용
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

            # 첫 번째 정답만 사용
            answers = examples["answers"]
            tokenized["labels"] = []

            for i, offsets in enumerate(offset_mapping):
                sample_idx = sample_mapping[i]
                gold_answers = answers[sample_idx]["text"]
                if len(gold_answers) == 0:
                    gold_answers = [""]

                with tokenizer.as_target_tokenizer():
                    label_enc = tokenizer(gold_answers[0], max_length=args.max_answer_length, truncation=True)
                tokenized["labels"].append(label_enc["input_ids"])

            # offset_mapping 보존
            # (start/end logits 시, postprocessing에 필요)
            # 여기서는 seq2seq 디코딩만 쓰므로 실제론 크게 안 쓰임
            tokenized["offset_mapping"] = offset_mapping
            tokenized["example_id"] = []
            for i in range(len(offset_mapping)):
                tokenized["example_id"].append(examples["id"][sample_mapping[i]])

            return tokenized

        def compute_metrics(eval_preds):
            """
            doc_stride 기반 chunk로부터 seq2seq 디코딩 결과 => SQuAD 스코어. 
            (start/end logits 없이 간단화)
            """
            preds, labels = eval_preds
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # SQuAD 형식에 맞춰서
            pred_list = [{"id": str(i), "prediction_text": p} for i, p in enumerate(decoded_preds)]
            ref_list = [{"id": str(i), "answers": {"text": [l], "answer_start": [0]}} for i, l in enumerate(decoded_labels)]

            result = squad_metric.compute(predictions=pred_list, references=ref_list)
            return result

    else:
        raise ValueError(f"Unsupported task: {task}")

    # ---------------------------------------------------------
    # Map 전처리
    # ---------------------------------------------------------
    # 'train' split의 column 이름을 제거
    remove_cols = dataset["train"].column_names
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_cols,
        num_proc=1,
    )

    if "test" not in tokenized_dataset:
        tokenized_dataset["test"] = tokenized_dataset["validation"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 1 epoch당 2번 eval
    eval_steps = max(1, len(tokenized_dataset["train"]) // (args.per_device_train_batch_size * 2))

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
        save_total_limit=1,  # 체크포인트 개수 제한
    )

    generation_kwargs = {
        "min_length": args.gen_min_length,
        "max_length": args.gen_max_length,
        "no_repeat_ngram_size": args.gen_no_repeat_ngram_size,
        "num_beams": args.gen_num_beams,
    }

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # RL 활성화 콜백
    trainer.add_callback(RLActivationCallback(args.do_RL, args.RL_start_epoch))
    # 5 epoch마다 test 셋 예측 콜백
    test_callback = TestEvaluationCallback(
        tokenized_dataset["test"], compute_metrics, tokenizer, task, generation_kwargs, output_dir
    )
    test_callback.trainer = trainer
    trainer.add_callback(test_callback)

    # ---------------------------------------------------------
    # 학습
    # ---------------------------------------------------------
    trainer.train()
    trainer.save_model(output_dir)

    # (추가) 원한다면 여기서 최종 test 예측 + 저장 가능
    # test_results = trainer.predict(tokenized_dataset["test"], **generation_kwargs)
    # ...
    print("Training done. Model & results saved.")


if __name__ == "__main__":
    main()
