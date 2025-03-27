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


# ---------------------------------------------------------
# Custom Trainer (추가 손실들을 wandb에 로깅)
# ---------------------------------------------------------
class CustomQuestionAnsweringSeq2SeqTrainer(QuestionAnsweringSeq2SeqTrainer):
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
    def __init__(self, test_dataset, dataset, compute_metrics, tokenizer, task, generation_kwargs, output_dir="results"):
        self.test_dataset = test_dataset
        self.dataset = dataset
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
        test_results = self.trainer.predict(self.test_dataset, self.dataset, **self.generation_kwargs)
        test_metrics = test_results.metrics
        
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
    # RL 관련 인자들
    parser.add_argument("--do_RL", action="store_true", default=False, help="Use Reinforcement Learning")
    parser.add_argument("--RL_expert_change_ratio", type=float, default=0.1, help="Expert change ratio")
    parser.add_argument("--RL_sample_num", type=int, default=4, help="Number of samples for RL")
    parser.add_argument("--RL_loss_coef", type=float, default=1.0, help="RL loss coefficient")
    parser.add_argument("--RL_sample_stretegy", type=str, default="multinomial", help="RL sample strategy", choices=["multinomial", "random"])
    parser.add_argument("--RL_base_logit_type", type=str, default="top1", help="RL base logit type", choices=["top1", "mean"])
    parser.add_argument("--RL_reward_stretegy", type=str, default="minus", help="RL reward strategy", choices=["minus", "static", "positive", "clamp", "log"])
    parser.add_argument("--use_sample_lm_loss", action="store_true", default=False, help="Use sample LM loss in RL")
    parser.add_argument("--RL_start_epoch", type=int, default=0)
    parser.add_argument("--RL_algo", default="reinforce", help="RL type", choices=["reinforce", "ppo"])
    parser.add_argument("--RL_ppo_eps", type=float, default=0.2, help="RL PPO epsilon")
    # Generation 인자 추가
    parser.add_argument("--gen_min_length", type=int, default=1, help="Minimum generation length")
    parser.add_argument("--gen_max_length", type=int, default=64, help="Maximum generation length")
    parser.add_argument("--gen_no_repeat_ngram_size", type=int, default=5, help="No repeat ngram size")
    parser.add_argument("--gen_num_beams", type=int, default=6, help="Number of beams for generation")
    # source_prefix 인자 추가 (summarization 전처리 시 사용)
    parser.add_argument("--source_prefix", type=str, default=None, help="Source prefix to prepend to input text for summarization")
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
        
    EPOCH = 0
    exp_name = f"{args.dataset_name}-{args.model_name.replace('/', '-')}-{task}-{args.num_train_epochs}epochs_final"
    if args.do_RL:
        run_name_parts = ["RL", args.RL_sample_stretegy, f"exp{args.RL_expert_change_ratio}", f"num{args.RL_sample_num}",
                          f"coef{args.RL_loss_coef}", args.RL_base_logit_type, args.RL_reward_stretegy, f"startRL{args.RL_start_epoch}"]
        if args.use_sample_lm_loss:
            run_name_parts.append("samplm")
        args.run_name += '_' + '_'.join(run_name_parts)


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
    print(model)
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
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
    
    if "test" in dataset.keys():
        tokenized_dataset["test"] = dataset["test"].map(preprocess_validation_function, batched=True, remove_columns=dataset["test"].column_names, num_proc=8)
    else:
        dataset["test"] = dataset["validation"]
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
    trainer = CustomQuestionAnsweringSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        eval_examples=dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        post_process_function=post_processing_function,
    )
    
    test_callback = TestEvaluationCallback(tokenized_dataset["test"], dataset["test"], compute_metrics, tokenizer, task, generation_kwargs, output_dir)
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
    test_results = trainer.predict(tokenized_dataset["test"], dataset["test"], **generation_kwargs)
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
