import wandb
import pandas as pd
import matplotlib.pyplot as plt

# 1. 필터 조건: 모든 조건을 만족하는 실험만 포함
# title = "SAMSUM"
# ENTITY = "isnlp_lab"
# PROJECT = "samsum-google-switch-base-8-summarization-15epochs_final"
# config_filter = {
#     "RL_loss_coef": 2.0,
#     "use_sample_lm_loss": False,
# }
# x_key = "RL_expert_change_ratio"
# x_key_name = x_key.split("_")[-1]
# end_key = "best_test_rouge2"
# get_type = "last"
# measurement_value = "test_rouge2"
# measurement_value_name = measurement_value.split("_")[-1]

title = "WIKI-2"
ENTITY = "isnlp_lab"
PROJECT = "wikitext-2-gpt2-text_generation-15epochs_final"
config_filter = {
    "RL_loss_coef": 2.0,
    "use_sample_lm_loss": True,
}
x_key = "RL_expert_change_ratio"
x_key_name = x_key.split("_")[-1]
end_key = "best_test_perplexity"
get_type = "min"
measurement_value = "test_perplexity"
measurement_value_name = measurement_value.split("_")[-1]

# 3. W&B API 접속
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

# 4. config 필터링 함수
def match_config(run_config, filter_dict):
    return all(run_config.get(k) == v for k, v in filter_dict.items())

# 5. 최고 test_rouge2 및 x_key 값 수집
records = []
for run in runs:
    try:
        if not match_config(run.config, config_filter):
            continue
        if x_key not in run.config:
            continue
        if end_key not in run.summary:
            continue  # ← 없으면 스킵
        
        history = run.history(keys=[measurement_value], pandas=True)
        if history.empty:
            continue
        print(f"Processed run: {run.name}\n x_value:{run.config[x_key]}")
        x_value = run.config[x_key]
        if get_type == "best":
            max_score = history[measurement_value].max()
        elif get_type == "last":
            max_score = list(history[measurement_value])[-1]
        elif get_type == "min":
            max_score = history[measurement_value].min()
        records.append({"x_value": x_value, "max_score": max_score})
    except Exception as e:
        print(f"Error with run {run.name}: {e}")

# 6. DataFrame 생성 및 평균/최대 정리 (동일 x_value 여러 run 있을 수 있음)
df = pd.DataFrame(records)

# scatter plot (겹치는 값 유지)
plt.figure(figsize=(10, 5))
plt.scatter(df["x_value"], df["max_score"], alpha=0.7)  # ← FIXED
plt.xlabel(x_key_name)
plt.ylabel(measurement_value_name)
plt.title(title)
plt.grid(True)
plt.tight_layout()
plt.savefig("tmp_plot.png")

# group-by plot (겹치는 값 제거 후 max/min 사용)
if get_type == "best":
    df_grouped = df.groupby("x_value")["max_score"].max().reset_index().sort_values("x_value")
elif get_type == "min":
    df_grouped = df.groupby("x_value")["max_score"].min().reset_index().sort_values("x_value")
plt.figure(figsize=(10, 5))
plt.plot(df_grouped["x_value"], df_grouped["max_score"], marker="o")
plt.xlabel(x_key_name)
plt.ylabel(measurement_value_name)
plt.title(title)
plt.grid(True)
plt.tight_layout()
plt.savefig("tmp.png")
