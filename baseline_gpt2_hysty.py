#!/usr/bin/env python
# coding=utf-8
"""
GPT‑2 LM 파인튜닝 스크립트 (MOE 적용 및 CustomTrainer/평가 콜백 포함)

- 데이터셋은 Salesforce/wikitext의 raw-v1 버전을 사용합니다.
- 입력 텍스트를 토큰화한 후, 전체 토큰을 하나로 이어 붙여 block_size 단위로 분할합니다.
- 모델 구성에 MOE 관련 파라미터를 추가하며, GPT‑2에 맞게 causal LM 학습을 진행합니다.
- CustomTrainer를 사용하여 추가 loss 항목들을 로그에 기록하며, 평가 콜백에서는 eval() 결과로 펄플렉시티를 계산합니다.
"""

import os
import sys
import math
import json
import wandb
import nltk
import numpy as np
from itertools import chain
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    AutoConfig,
    HfArgumentParser,
    set_seed,
)
from base_GPT2 import GPT2LMHeadModel  # 사용자 정의 GPT2LMHeadModel (MOE 관련 to_moe() 포함)

# nltk 문장 토크나이저 다운로드 (없으면)
try:
    nltk.download('punkt_tab')
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ========================
# 인자 정의 (Model, Data, Training)
# ========================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="gpt2", metadata={"help": "학습할 모델의 이름 또는 경로"}
    )
    moe_config_path: str = field(
        default="google/switch-base-8", metadata={"help": "MOE 설정 파일 (모델 구성) 경로"}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="wikitext-2", metadata={"help": "불러올 데이터셋 이름"}
    )
    block_size: int = field(
        default=1024, metadata={"help": "토큰화 후 분할할 블록 크기 (최대 입력 길이)"}
    )
    preprocessing_num_workers: int = field(
        default=4, metadata={"help": "토크나이징에 사용할 프로세스 수"}
    )

# ========================
# CustomTrainer 정의 (추가 loss 항목 로깅)
# ========================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        # 기본 loss 또는 첫번째 리턴값 사용
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

# ========================
# 평가 콜백 (펄플렉시티 계산)
# ========================
class TestEvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, compute_metrics, tokenizer, task, output_dir="results"):
        self.test_dataset = test_dataset
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.task = task
        self.trainer = None
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        print("-- Training begins at epoch", state.epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        print("#" * 50)
        print(f"Epoch {state.epoch} train end")
        print("#" * 50)
        # 매 5 에폭마다 평가 수행
        if int(state.epoch) % 5 != 0:
            return control

        eval_results = self.trainer.evaluate(self.test_dataset)
        eval_loss = eval_results.get("eval_loss")
        perplexity = math.exp(eval_loss) if eval_loss is not None else None
        test_metrics = {"perplexity": perplexity}
        results_file = os.path.join(self.output_dir, f"{state.epoch}_results.json")
        with open(results_file, "w") as f:
            json.dump({k: round(v, 4) for k, v in test_metrics.items()}, f, indent=4)
        print("Evaluation results:", test_metrics)
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        return control

# ========================
# 데이터 전처리 함수: 토큰화 및 그룹핑 (청크 분할)
# ========================
def tokenize_function(examples, tokenizer):
    # 각 입력 문장을 토큰화 (batch["text"]가 리스트라고 가정)
    return tokenizer(examples["text"], truncation=True)

def group_texts(examples, block_size):
    # 예제에서 생성된 token lists를 하나의 긴 리스트로 결합
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    # 총 길이를 block_size로 나눈 배수로 자르기
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# ========================
# 메인 함수
# ========================
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 랜덤 시드 설정
    set_seed(training_args.seed)

    # wandb 초기화
    exp_name = f"{data_args.dataset_name}-{model_args.model_name_or_path.replace('/', '-')}-lm-{training_args.num_train_epochs}epochs"
    output_dir = os.path.join("results", exp_name, training_args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    wandb.init(project=exp_name, name=training_args.run_name)

    # 데이터셋 로드 (Salesforce/wikitext의 raw-v1 사용)
    dataset = load_dataset("Salesforce/wikitext", name=f"{data_args.dataset_name}-raw-v1")
    print("Loaded dataset:", dataset)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if model_args.model_name_or_path == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    # 토큰화 수행 (train과 validation 모두)
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Tokenizing dataset",
    )

    # 그룹핑: 전체 텍스트를 하나로 이어 붙여 block_size 단위로 나눔
    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, data_args.block_size),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {data_args.block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"] if "validation" in lm_datasets else None
    # test셋이 없다면 validation 사용
    test_dataset = eval_dataset

    # ------------------------------
    # 모델 및 MOE 설정
    # ------------------------------
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    moe_config = AutoConfig.from_pretrained(model_args.moe_config_path)
    model_config.num_experts = moe_config.num_experts
    model_config.expert_capacity = moe_config.expert_capacity
    model_config.router_bias = moe_config.router_bias
    model_config.router_jitter_noise = moe_config.router_jitter_noise
    model_config.router_dtype = moe_config.router_dtype
    model_config.router_ignore_padding_tokens = moe_config.router_ignore_padding_tokens
    model_config.router_z_loss_coef = moe_config.router_z_loss_coef
    model_config.router_aux_loss_coef = moe_config.router_aux_loss_coef

    model = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        device_map="auto",
    )
    model.to_moe()  # 사용자 정의 함수: MOE 적용
    print(model)

    # ------------------------------
    # Data Collator (동적 패딩)
    # ------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    # ------------------------------
    # 평가 지표: eval_loss를 기반으로 펄플렉시티 계산
    # ------------------------------
    def compute_metrics(eval_preds):
        eval_loss = eval_preds.metrics.get("eval_loss")
        perplexity = math.exp(eval_loss) if eval_loss is not None else None
        return {"perplexity": perplexity}

    # ------------------------------
    # Trainer 생성 (CustomTrainer 사용)
    # ------------------------------
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 평가 콜백 추가
    test_callback = TestEvaluationCallback(test_dataset, compute_metrics, tokenizer, "text_generation", output_dir)
    test_callback.trainer = trainer
    trainer.add_callback(test_callback)

    # 학습 시작
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
