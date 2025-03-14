import wandb
import pandas as pd

# wandb Api 객체 생성
api = wandb.Api()

# 1) 원하는 entity와 project 이름으로 모든 runs 불러오기
runs = api.runs("samsum-google-switch-base-8")

run_data = []
for run in runs:
    # run의 config(학습 때 argparse로 넘긴 값들)
    config = run.config
    
    # run의 summary(최종 메트릭 등)
    summary = run.summary
    
    # 예) test셋의 ROUGE를 run.summary에 기록했다고 가정
    test_rouge1 = summary.get("test/rouge1", None)
    test_rouge2 = summary.get("test/rouge2", None)
    test_rougeL = summary.get("test/rougeL", None)
    
    # 여기서 필요한 RL 파라미터 추출
    # (실제로 wandb에 기록되는 이름이 argparse 인자와 동일한지 확인 필요)
    do_RL = config.get("do_RL", None)
    RL_expert_change_ratio = config.get("RL_expert_change_ratio", None)
    RL_sample_num = config.get("RL_sample_num", None)
    RL_loss_coef = config.get("RL_loss_coef", None)
    RL_sample_stretegy = config.get("RL_sample_stretegy", None)
    RL_base_logit_type = config.get("RL_base_logit_type", None)
    RL_reward_stretegy = config.get("RL_reward_stretegy", None)
    use_sample_lm_loss = config.get("use_sample_lm_loss", None)
    RL_start_epoch = config.get("RL_start_epoch", None)
    RL_algo = config.get("RL_algo", None)
    RL_ppo_eps = config.get("RL_ppo_eps", None)
    
    run_data.append({
        "run_name": run.name,
        "do_RL": do_RL,
        "RL_expert_change_ratio": RL_expert_change_ratio,
        "RL_sample_num": RL_sample_num,
        "RL_loss_coef": RL_loss_coef,
        "RL_sample_stretegy": RL_sample_stretegy,
        "RL_base_logit_type": RL_base_logit_type,
        "RL_reward_stretegy": RL_reward_stretegy,
        "use_sample_lm_loss": use_sample_lm_loss,
        "RL_start_epoch": RL_start_epoch,
        "RL_algo": RL_algo,
        "RL_ppo_eps": RL_ppo_eps,
        "test_rouge1": test_rouge1,
        "test_rouge2": test_rouge2,
        "test_rougeL": test_rougeL,
    })

# DataFrame 생성
df = pd.DataFrame(run_data)

# groupby 기준: RL 파라미터들을 전부 넣음
group_cols = [
    "do_RL", 
    "RL_expert_change_ratio", 
    "RL_sample_num", 
    "RL_loss_coef",
    "RL_sample_stretegy",
    "RL_base_logit_type",
    "RL_reward_stretegy",
    "use_sample_lm_loss",
    "RL_start_epoch",
    "RL_algo",
    "RL_ppo_eps"
]

# 계산할 메트릭들
metrics = ["test_rouge1", "test_rouge2", "test_rougeL"]

# groupby 후, 각 metric에 대해 평균
grouped_df = df.groupby(group_cols)[metrics].mean().reset_index()

# 결과 출력 (필요하면 csv로 저장 가능)
print(grouped_df)
# grouped_df.to_csv("wandb_RL_summary.csv", index=False)


for param in group_cols:
    grouped = df.groupby(param)[metrics].mean().reset_index()
    print(f"\n=== Grouped by '{param}' ===")
    print(grouped)
