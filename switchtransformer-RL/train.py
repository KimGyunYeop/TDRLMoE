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
from Custom_MoE_SwitchT import SwitchTransformersForConditionalGeneration, SwitchTransformersConfig

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
        return control
# ---------------------------------------------------------
# RL Activation Callback: 각 에폭 시작 시 RL_start_epoch 기준으로 RL 활성화 여부 결정
# ---------------------------------------------------------
class RLActivationCallback(TrainerCallback):
    def __init__(self, do_RL, RL_start_epoch):
        self.do_RL = do_RL
        self.RL_start_epoch = RL_start_epoch

    def on_epoch_begin(self, args, state, control, **kwargs):
        # 모델 인스턴스 가져오기 (kwargs 또는 self.trainer에서)
        model = kwargs.get("model")
        if model is None and hasattr(self, "trainer"):
            model = self.trainer.model
        if self.do_RL:
            if state.epoch >= self.RL_start_epoch:
                model.config.do_RL = True
                model.do_RL = True
                print(f"Epoch {state.epoch:.2f}: RL 활성화 (config.do_RL={model.config.do_RL}, do_RL={model.do_RL})")
            else:
                model.config.do_RL = False
                model.do_RL = False
                print(f"Epoch {state.epoch:.2f}: RL 비활성화 (do_RL={model.config.do_RL}, do_RL={model.do_RL})")
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
                        default="samsum", help="Dataset name to load")
    # NLU 전용: glue, superglue의 세부 태스크 이름 (없으면 기본값 사용)
    parser.add_argument("--nlu_task", type=str, default=None, help="For glue/superglue, specify task name (e.g., sst2 for glue, boolq for superglue)")
    
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
    parser.add_argument("--gen_max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5, help="No repeat ngram size")
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

    EPOCH = 0
    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs_final"
    if args.do_RL:
        run_name_parts = ["RL", args.RL_sample_stretegy, f"exp{args.RL_expert_change_ratio}", f"num{args.RL_sample_num}",
                          f"coef{args.RL_loss_coef}", args.RL_base_logit_type, args.RL_reward_stretegy, f"startRL{args.RL_start_epoch}"]
        if args.use_sample_lm_loss:
            run_name_parts.append("samplm")
        args.run_name += '_' + '_'.join(run_name_parts)

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

    # 초기 RL 상태는 RL_start_epoch에 따라 설정 (첫 에폭 시작 전 설정)
    if args.do_RL and 0 >= args.RL_start_epoch:
        model.config.do_RL = True
        model.do_RL = True
        print("초기 RL 활성화")
    else:
        model.config.do_RL = False
        model.do_RL = False
        print("초기 RL 비활성화")

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

        best_metric = "rouge2"
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
        best_metric = "accuracy"
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
        best_metric = "f1"
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

    # Map 전처리 함수 (각 split에 대해)
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names, num_proc=8)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    if "test" not in tokenized_dataset.keys():
        tokenized_dataset["test"] = tokenized_dataset["validation"]
        

    # ---------------------------------------------------------
    # Training Arguments 설정
    # ---------------------------------------------------------
    eval_steps = len(tokenized_dataset["train"]) // (args.per_device_train_batch_size)
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
        load_best_model_at_end=True,               # 베스트 모델 자동 불러오기 활성화
        metric_for_best_model=best_metric,          # 평가 지표 지정
        greater_is_better=True                     # 낮은 eval_loss가 좋은 모델임을 지정
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
    # RL 활성화 상태를 각 에폭 시작 시 업데이트하는 콜백 추가
    trainer.add_callback(RLActivationCallback(args.do_RL, args.RL_start_epoch))
    test_callback = TestEvaluationCallback(tokenized_dataset["test"], compute_metrics, tokenizer, task, generation_kwargs, output_dir)
    test_callback.trainer = trainer  # trainer 인스턴스를 직접 할당
    trainer.add_callback(test_callback)

    # ---------------------------------------------------------
    # 모델 학습 및 평가
    # ---------------------------------------------------------
    trainer.train()
    trainer.save_model(output_dir)
    
    # 최종 테스트 예측 시 generation 인자 전달
    test_results = trainer.predict(tokenized_dataset["test"], **generation_kwargs)
    predictions = test_results.predictions
    labels = test_results.label_ids

    if task in ["summarization", "qa", "nlu", "translation"]:
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
    
    print(f"Test metrics at epoch of Best Model: {final_metrics}")
    wandb.log({f"best_test_{k}": v for k, v in final_metrics.items()})

if __name__ == "__main__":
    main()
