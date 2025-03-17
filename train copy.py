import argparse
import math
import numpy as np
import json
import wandb
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from Custom_MoE3 import SwitchTransformersForConditionalGeneration, SwitchTransformersConfig
import os
import torch
from transformers import Seq2SeqTrainer

# ---------------------------------------------------------
# Custom Trainer (추가 손실들을 wandb에 로깅)
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
            "sample_lm_loss"
        ]:
            val = getattr(outputs, loss_name, None)
            if val is not None:
                log_dict[loss_name] = val.detach().float().mean().item()
        if self.state.global_step % self.args.logging_steps == 0:
            self.log(log_dict)
        return (loss, outputs) if return_outputs else loss

# ---------------------------------------------------------
# 5 epoch마다 테스트셋을 평가하기 위한 콜백
# ---------------------------------------------------------
class TestEveryNEpochsCallback(TrainerCallback):
    """
    n Epoch마다 테스트셋에 대해 predict를 수행하는 콜백.
    """
    def __init__(self, test_dataset, n, args, tokenizer, compute_metrics, task):
        """
        :param test_dataset: 전처리된 테스트셋 (tokenized_dataset["test"])
        :param n: 몇 epoch마다 테스트를 돌릴지 (예: 5)
        :param args: 학습/추론 인자 (gen_max_length, gen_min_length, etc.)
        :param tokenizer: 모델 tokenizer
        :param compute_metrics: 테스트셋 metric 계산 함수
        :param task: 현재 태스크 (summarization, nlu 등)
        """
        self.test_dataset = test_dataset
        self.n = n
        self.args = args
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.task = task
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", None)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        epoch = state.epoch
        if epoch is None:
            return control

        current_epoch = int(epoch)
        # epoch가 0이 아니고, n으로 나누어떨어지면 테스트
        if current_epoch != 0 and (current_epoch % self.n == 0):
            print(f"\n*** Running test evaluation at epoch {current_epoch} ***\n")
            # 테스트셋 predict (hyper_MoE 스타일 generation 파라미터 사용)
            test_results = self.trainer.predict(
                self.test_dataset,
                max_length=self.args.gen_max_length,
                min_length=self.args.gen_min_length,
                no_repeat_ngram_size=self.args.gen_no_repeat_ngram_size,
                num_beams=self.args.gen_num_beams,
                length_penalty=self.args.length_penalty
            )
            predictions = test_results.predictions
            labels = test_results.label_ids

            if self.task in ["summarization", "qa", "nlu"]:
                preds = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            else:
                decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            if self.task == "text_generation":
                test_metrics = {}
            else:
                test_metrics = self.compute_metrics((predictions, labels))

            print(f"Test metrics at epoch {current_epoch}: {test_metrics}")
            wandb.log({f"test_epoch_{current_epoch}_{k}": v for k, v in test_metrics.items()})

        return control

# ---------------------------------------------------------
# (선택) 요약문 후처리를 위한 함수 (ROUGE 계산 전 tokenization 정리)
# ---------------------------------------------------------
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def postprocess_text(preds, labels):
    """
    예측문/정답문에서 strip 후, nltk.sent_tokenize로 문장 단위 분할 -> 줄바꿈 추가
    """
    str_preds = [pred.strip() for pred in preds]
    str_labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in str_preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in str_labels]
    return preds, labels, str_preds, str_labels

# ---------------------------------------------------------
# Argument parsing
# (hyper_MoE 스타일 인자들 추가)
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Switch Transformer (TDLRMoE) with possible RL, plus hyper_MoE-like preprocessing & test-eval logic"
    )
    # -----------------------------------------------------
    # 기존 인자 (RL 등)
    # -----------------------------------------------------
    parser.add_argument("--model_name", type=str, default="google/switch-base-16", help="HuggingFace model identifier")
    parser.add_argument("--dataset_name", type=str,
                        choices=["samsum","openwebtext","wikitext-2","wikitext-103","glue","superglue","squad_v1","xsum","cnn_dailymail"],
                        default="samsum", help="Dataset name to load")
    parser.add_argument("--nlu_task", type=str, default=None, help="For glue/superglue, specify task name")
    parser.add_argument("--source_lang", type=str, default=None, help="Source language code (if needed)")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language code (if needed)")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (train)")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="Grad accumulation steps")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (eval)")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default="run", help="Wandb run name")

    # RL 관련 인자
    parser.add_argument("--do_RL", action="store_true", default=False, help="Use Reinforcement Learning")
    parser.add_argument("--RL_expert_change_ratio", type=float, default=0.1, help="Expert change ratio")
    parser.add_argument("--RL_sample_num", type=int, default=4, help="Number of samples for RL")
    parser.add_argument("--RL_loss_coef", type=float, default=1.0, help="RL loss coefficient")
    parser.add_argument("--RL_sample_stretegy", type=str, default="multinomial", help="RL sample strategy",
                        choices=["multinomial", "random"])
    parser.add_argument("--RL_base_logit_type", type=str, default="top1", help="RL base logit type",
                        choices=["top1","mean"])
    parser.add_argument("--RL_reward_stretegy", type=str, default="minus", help="RL reward strategy",
                        choices=["minus","static","positive","clamp"])
    parser.add_argument("--use_sample_lm_loss", action="store_true", default=False, help="Use sample LM loss in RL")
    parser.add_argument("--RL_start_epoch", type=int, default=0)
    parser.add_argument("--RL_algo", default="reinforce", help="RL type", choices=["reinforce","ppo"])
    parser.add_argument("--RL_ppo_eps", type=float, default=0.2, help="RL PPO epsilon")

    parser.add_argument("--max_source_length", type=int, default=1024, help="Max source length (like hyper_MoE)")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max target length (like hyper_MoE)")
    parser.add_argument("--ignore_pad_token_for_loss", action="store_true", default=False,
                        help="Replace PAD with -100 to ignore them in loss.")
    parser.add_argument("--gen_min_length", type=int, default=10, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=60, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=3, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")

    parser.add_argument("--evaluation_strategy", type=str, default="epoch",
                        help="evaluation_strategy: 'epoch' to mimic hyper_MoE")
    parser.add_argument("--save_strategy", type=str, default="no", 
                        help="Save strategy: 'no' to mimic hyper_MoE (often no checkpoint saves)")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay=0.1 to mimic hyper_MoE")

    # '5 epoch마다 test'를 위한 인자
    parser.add_argument("--test_eval_interval", type=int, default=5,
                        help="Run test evaluation every N epochs. Default=5")

    return parser.parse_args()

def main():
    args = parse_args()

    # 태스크 결정
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
    if args.do_RL:
        run_name_parts = ["RL", args.RL_sample_stretegy, f"exp{args.RL_expert_change_ratio}", f"num{args.RL_sample_num}",
                          f"coef{args.RL_loss_coef}", args.RL_base_logit_type, args.RL_reward_stretegy, f"startRL{args.RL_start_epoch}"]
        if args.use_sample_lm_loss:
            run_name_parts.append("samplm")
        args.run_name += "_" + "_".join(run_name_parts)

    if os.path.exists(os.path.join("results", exp_name, args.run_name, "pred_gold_samples.json")):
        print("Results already exist. Skipping training.")
        return

    output_dir = os.path.join("results", exp_name, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(project=exp_name, name=args.run_name)

    # ---------------------------------------------------------
    # 데이터셋 로드
    # ---------------------------------------------------------
    if args.dataset_name == "samsum":
        dataset = load_dataset("samsum")
    elif args.dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext")
    elif args.dataset_name in ["wikitext-2", "wikitext-103"]:
        dataset = load_dataset(args.dataset_name)
    elif args.dataset_name == "glue":
        tname = args.nlu_task if args.nlu_task else "sst2"
        dataset = load_dataset("glue", tname)
    elif args.dataset_name == "superglue":
        tname = args.nlu_task if args.nlu_task else "boolq"
        dataset = load_dataset("superglue", tname)
    elif args.dataset_name == "squad_v1":
        dataset = load_dataset("squad")
    elif args.dataset_name == "xsum":
        dataset = load_dataset("xsum", trust_remote_code=True)
    elif args.dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", "3.0.0")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    print(dataset)

    # ---------------------------------------------------------
    # 모델 및 토크나이저 로드
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = SwitchTransformersConfig.from_pretrained(args.model_name)
    # RL 인자 주입
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
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        config=model_config,
        device_map="auto"
    )

    # ---------------------------------------------------------
    # 전처리 & 평가 지표
    #   - 아래서 max_source_length, max_target_length,
    #     padding='max_length' 등에 대한 처리
    # ---------------------------------------------------------
    if task == "summarization":
        # summarization용 전처리
        def preprocess_function(examples):
            # (필요 시 prefix = "summarize: " + ...
            # 여기서는 cnn_dailymail, xsum, samsum 등 처리)
            if "dialogue" in examples and "summary" in examples:  # samsum
                inputs = examples["dialogue"]
                targets = examples["summary"]
            elif "document" in examples and "summary" in examples:  # xsum
                inputs = examples["document"]
                targets = examples["summary"]
            elif "article" in examples and "highlights" in examples:  # cnn_dailymail
                inputs = examples["article"]
                targets = examples["highlights"]
            else:
                # fallback
                in_col = list(examples.keys())[0]
                out_col = list(examples.keys())[1]
                inputs = examples[in_col]
                targets = examples[out_col]

            # 입력
            model_inputs = tokenizer(
                inputs,
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True
            )
            # 라벨
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=args.max_target_length,
                    padding="max_length",
                    truncation=True
                )
            if args.ignore_pad_token_for_loss:
                label_ids = []
                for seq in labels["input_ids"]:
                    seq = [(l if l != tokenizer.pad_token_id else -100) for l in seq]
                    label_ids.append(seq)
                labels["input_ids"] = label_ids

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # ROUGE
        rouge_metric = evaluate.load("rouge")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # postprocess_text 적용 (문장 단위로 잘라줄 때)
            decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)

            result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {k: round(v * 100, 4) for k,v in result.items()}
            return result

    elif task == "text_generation":
        def preprocess_function(examples):
            # openwebtext, wikitext
            col = "text"
            if col not in examples:
                col = list(examples.keys())[0]
            inputs = tokenizer(
                examples[col],
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True
            )
            # LM이므로 inputs == labels
            if args.ignore_pad_token_for_loss:
                # pad -> -100
                label_ids = []
                for seq in inputs["input_ids"]:
                    seq = [(l if l != tokenizer.pad_token_id else -100) for l in seq]
                    label_ids.append(seq)
                inputs["labels"] = label_ids
            else:
                inputs["labels"] = inputs["input_ids"].copy()
            return inputs

        def compute_metrics(eval_preds):
            # 텍스트 생성은 보통 ppl
            return {}

    elif task == "nlu":
        # glue/superglue
        def preprocess_function(examples):
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
                    label_ids = []
                    for seq in labels["input_ids"]:
                        seq = [(l if l != tokenizer.pad_token_id else -100) for l in seq]
                        label_ids.append(seq)
                    labels["input_ids"] = label_ids
                inputs["labels"] = labels["input_ids"]
            else:
                inputs["labels"] = None
            return inputs

        acc_metric = evaluate.load("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds = [p.strip() for p in decoded_preds]
            decoded_labels = [l.strip() for l in decoded_labels]
            result = acc_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return result

    elif task == "qa":
        # squad v1
        def preprocess_function(examples):
            inputs = tokenizer(
                examples["question"], examples["context"],
                max_length=args.max_source_length,
                padding="max_length",
                truncation=True
            )
            answers = [a[0] if len(a)>0 else "" for a in examples["answers"]["text"]]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    answers,
                    max_length=args.max_target_length,
                    padding="max_length",
                    truncation=True
                )
            if args.ignore_pad_token_for_loss:
                label_ids = []
                for seq in labels["input_ids"]:
                    seq = [(l if l != tokenizer.pad_token_id else -100) for l in seq]
                    label_ids.append(seq)
                labels["input_ids"] = label_ids
            inputs["labels"] = labels["input_ids"]
            return inputs

        squad_metric = evaluate.load("squad")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            result = squad_metric.compute(predictions=decoded_preds, references=decoded_labels)
            return result

    # ---------------------------------------------------------
    # 데이터셋 전처리
    # ---------------------------------------------------------
    column_names = dataset["train"].column_names
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names
    )

    # ---------------------------------------------------------
    # Data Collator
    # ---------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ---------------------------------------------------------
    # TrainingArguments
    # ---------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy=args.evaluation_strategy,  # 기본 epoch
        save_strategy=args.save_strategy,              # 기본 no
        weight_decay=args.weight_decay,                # 기본 0.1
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        predict_with_generate=False,
        run_name=args.run_name,
        report_to=["wandb"],
        seed=args.seed,
        fp16=args.fp16,
        save_total_limit=1
    )

    # ---------------------------------------------------------
    # Trainer 생성 (CustomSeq2SeqTrainer)
    # ---------------------------------------------------------
    # Validation 셋이 없을 수 있으니 예외처리
    if "validation" in tokenized_dataset:
        eval_dataset = tokenized_dataset["validation"]
    else:
        eval_dataset = None

    test_dataset = None
    if "test" in tokenized_dataset:
        test_dataset = tokenized_dataset["test"]

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---------------------------------------------------------
    # 5 epoch마다 test 평가 콜백 등록
    # ---------------------------------------------------------
    if test_dataset is not None:
        test_callback = TestEveryNEpochsCallback(
            test_dataset=test_dataset,
            n=args.test_eval_interval,
            args=args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            task=task
        )
        trainer.add_callback(test_callback)

    # ---------------------------------------------------------
    # 학습 & 최종 평가
    # ---------------------------------------------------------
    trainer.train()
    trainer.save_model(output_dir)

    # (선택) 최종 테스트셋 평가
    if test_dataset is not None:
        final_test_result = trainer.predict(
            test_dataset,
            max_length=args.gen_max_length,
            min_length=args.gen_min_length,
            no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
            num_beams=args.gen_num_beams,
            length_penalty=args.length_penalty
        )
        predictions = final_test_result.predictions
        labels = final_test_result.label_ids

        if task in ["summarization", "qa", "nlu"]:
            preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        else:
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        if task == "text_generation":
            eval_loss = final_test_result.metrics.get("eval_loss")
            perplexity = math.exp(eval_loss) if eval_loss is not None else None
            final_metrics = {"perplexity": perplexity}
        else:
            final_metrics = compute_metrics((predictions, labels))

        sample_list = []
        for pred, gold in zip(decoded_preds, decoded_labels):
            sample_list.append({"prediction": pred, "gold": gold})
        with open(os.path.join(output_dir, "pred_gold_samples.json"), "w", encoding="utf-8") as f:
            json.dump(sample_list, f, indent=4, ensure_ascii=False)

        results_file = os.path.join(output_dir, f"{task}_switch_results.json")
        with open(results_file, "w") as f:
            json.dump({k: round(v, 4) for k, v in final_metrics.items()}, f, indent=4)
        print("Final Test results:", final_metrics)

    print("Training & evaluation completed.")

if __name__ == "__main__":
    main()
