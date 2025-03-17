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
    Seq2SeqTrainer,
    TrainerCallback,
)
from Custom_MoE3 import SwitchTransformersForConditionalGeneration, SwitchTransformersConfig

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
# TestEvaluationCallback 수정 (5 에폭마다 평가 및 generation 인자 전달, postprocessing 적용)
# ---------------------------------------------------------
class TestEvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, compute_metrics, tokenizer, task, generation_kwargs):
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.task = task
        self.trainer = None
        self.generation_kwargs = generation_kwargs

    def on_train_begin(self, args, state, control, **kwargs):
        # trainer 인스턴스를 저장합니다.
        self.trainer = kwargs.get("trainer", None)

    def on_epoch_end(self, args, state, control, **kwargs):
        # 5 에폭마다 평가 수행
        if int(state.epoch) % 5 != 0:
            return control

        if self.trainer is None:
            print("Trainer is not set in callback.")
            return control

        # predict 호출 시 generation 인자 전달
        test_results = self.trainer.predict(self.test_dataset, **self.generation_kwargs)
        predictions = test_results.predictions
        labels = test_results.label_ids

        if self.task in ["summarization", "qa", "nlu"]:
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
            test_metrics = {}
        else:
            test_metrics = self.compute_metrics((predictions, labels))
        
        print(f"Test metrics at epoch {state.epoch}: {test_metrics}")
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        return control

# ---------------------------------------------------------
# Argument parsing (dataset_name에 따라 자동으로 태스크 결정)
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Switch Transformer on multiple tasks: NLU, QA, Summarization, or Text Generation."
    )
    parser.add_argument("--model_name", type=str, default="google/switch-base-16", help="HuggingFace model identifier")
    parser.add_argument("--dataset_name", type=str, 
                        choices=["samsum", "openwebtext", "wikitext-2", "wikitext-103", "glue", "superglue", "squad_v1", "xsum", "cnn_dailymail"],
                        default="samsum", help="Dataset name to load")
    # NLU 전용: glue, superglue의 세부 태스크 이름 (없으면 기본값 사용)
    parser.add_argument("--nlu_task", type=str, default=None, help="For glue/superglue, specify task name (e.g., sst2 for glue, boolq for superglue)")
    # 번역 태스크가 아닌 경우 필요없으나 이전 코드 유지 (wmt23 등)
    parser.add_argument("--source_lang", type=str, default=None, help="Source language code (if needed)")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language code (if needed)")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device during evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X update steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X update steps")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default="run", help="Wandb run name")
    # RL 관련 인자들
    parser.add_argument("--do_RL", action="store_true", default=False, help="Use Reinforcement Learning")
    parser.add_argument("--RL_expert_change_ratio", type=float, default=0.1, help="Expert change ratio")
    parser.add_argument("--RL_sample_num", type=int, default=4, help="Number of samples for RL")
    parser.add_argument("--RL_loss_coef", type=float, default=1.0, help="RL loss coefficient")
    parser.add_argument("--RL_sample_stretegy", type=str, default="multinomial", help="RL sample strategy", choices=["multinomial", "random"])
    parser.add_argument("--RL_base_logit_type", type=str, default="top1", help="RL base logit type", choices=["top1", "mean"])
    parser.add_argument("--RL_reward_stretegy", type=str, default="minus", help="RL reward strategy", choices=["minus", "static", "positive", "clamp"])
    parser.add_argument("--use_sample_lm_loss", action="store_true", default=False, help="Use sample LM loss in RL")
    parser.add_argument("--RL_start_epoch", type=int, default=0)
    parser.add_argument("--RL_algo", default="reinforce", help="RL type", choices=["reinforce", "ppo"])
    parser.add_argument("--RL_ppo_eps", type=float, default=0.2, help="RL PPO epsilon")
    # Generation 인자 추가
    parser.add_argument("--gen_min_length", type=int, default=10, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=60, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=3, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    # source_prefix 인자 추가 (summarization 전처리 시 사용)
    parser.add_argument("--source_prefix", type=str, default="summarize: ", help="Source prefix to prepend to input text for summarization")
    return parser.parse_args()

# ---------------------------------------------------------
# Main 함수: 데이터셋 이름에 따라 태스크 및 전처리/평가 함수 결정
# ---------------------------------------------------------
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
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    args.task = task
    # 사용자가 별도로 source_prefix를 지정하지 않았다면 기본값을 사용
    if not hasattr(args, "source_prefix") or args.source_prefix is None:
        args.source_prefix = default_prefix

    # 번역 태스크 관련 체크 (wmt23 등)
    if task == "translation":
        if args.source_lang is None or args.target_lang is None:
            raise ValueError("For translation task, please specify both --source_lang and --target_lang.")

    EPOCH = 0
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
    # 데이터셋 로드 (태스크별 분기)
    # ---------------------------------------------------------
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
    print(dataset)

    # ---------------------------------------------------------
    # 모델 및 토크나이저 로드
    # ---------------------------------------------------------
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
    print(model_config)
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        config=model_config,
        device_map="auto"
    )
    if args.do_RL:
        if EPOCH >= args.RL_start_epoch:
            model.config.do_RL = True
            print("RL is activated")
        else:
            model.config.do_RL = False
            print("RL is deactivated")

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
        def preprocess_function(batch):
            # QA 태스크의 경우, 질문(question)에 prefix를 추가
            inputs = tokenizer([args.source_prefix + question for question in batch["question"]],
                            batch["context"], truncation=True)
            answers = [ans[0] if len(ans) > 0 else "" for ans in batch["answers"]["text"]]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(answers, truncation=True)
            inputs["labels"] = labels["input_ids"]
            return inputs

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

    else:
        raise ValueError(f"Unsupported task: {task}")

    # Map 전처리 함수 (각 split에 대해)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # ---------------------------------------------------------
    # 1에폭에 두 번씩 eval 수행하도록 eval_steps 설정
    # ---------------------------------------------------------
    eval_steps = len(tokenized_dataset["train"]) // (args.per_device_train_batch_size * 2)

    # ---------------------------------------------------------
    # Training Arguments 설정
    # ---------------------------------------------------------
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
        predict_with_generate=False,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        fp16=args.fp16,
        save_total_limit=1,
    )

    # Generation 인자 딕셔너리 생성
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
    # Callback에 generation 인자 전달
    trainer.add_callback(TestEvaluationCallback(tokenized_dataset["test"], compute_metrics, tokenizer, task, generation_kwargs))

    # ---------------------------------------------------------
    # 모델 학습 및 평가
    # ---------------------------------------------------------
    trainer.train()
    trainer.save_model(output_dir)
    
    # 최종 테스트 예측 시 generation 인자 전달
    test_results = trainer.predict(tokenized_dataset["test"], **generation_kwargs)
    predictions = test_results.predictions
    labels = test_results.label_ids

    if task in ["summarization", "qa", "nlu"]:
        preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # postprocessing 적용: 문장 단위 줄바꿈
        decoded_preds, decoded_labels, _, _ = postprocess_text(decoded_preds, decoded_labels)
    elif task in ["text_generation"]:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    if task == "text_generation":
        eval_loss = test_results.metrics.get("eval_loss")
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
    print("Model and results saved.")

if __name__ == "__main__":
    main()
