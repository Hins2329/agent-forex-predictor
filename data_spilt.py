import pandas as pd
from pathlib import Path

df = pd.read_csv("data/processed/state_minute.csv", parse_dates=["datetime"])

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_end = "2025-03-31"
predict_start = "2025-04-01"
predict_end = "2025-04-17"

train_df = df[df["datetime"] <= train_end]
train_df.to_csv(output_dir / "train_state.csv", index=False)
print(f"✅ Saved training data: {len(train_df)} rows")

predict_df = df[(df["datetime"] >= predict_start) & (df["datetime"] <= predict_end)].copy()
predict_df.drop(columns=["exchange_rate"], inplace=True)
predict_df.to_csv(output_dir / "test_state_for_predict.csv", index=False)
print(f"✅ Saved prediction input data: {len(predict_df)} rows")