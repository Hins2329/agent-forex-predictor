import pandas as pd

# === macro
macro = {
    "CPI": 319.615,
    "InterestRate": 4.33,
    "Unemployment": 4.2
}

# === loading news
news_df = pd.read_csv("data/processed/news_tone.csv", parse_dates=["datetime"])

# filter out 2025-04-18 news and take the average tone
tone_day = news_df[news_df["datetime"].dt.date == pd.to_datetime("2025-04-18").date()]
avg_tone = tone_day["tone"].mean() if not tone_day.empty else 0.0

# get the last data from 4/17
fx_df = pd.read_csv("data/processed/usd_cny_minute_twelvedata.csv", parse_dates=["datetime"])

fx_417 = fx_df[fx_df["datetime"].dt.date == pd.to_datetime("2025-04-17").date()]
fx_417 = fx_417.sort_values("datetime")
latest_price = fx_417.iloc[-1]["close"]  # 使用最后一条汇率数据作为预测基准

# build state (exchange rate of the day, macro, news tone)
state = {
    "datetime": pd.to_datetime("2025-04-18"),  # 预测日期
    "CPI": macro["CPI"],
    "InterestRate": macro["InterestRate"],
    "Unemployment": macro["Unemployment"],
    "tone": avg_tone,
    "exchange_rate": latest_price,
    "holding": 0
}

# saving for predict
df_state = pd.DataFrame([state])
df_state.to_csv("data/processed/state_20250418.csv", index=False)
print("✅ 已生成 4月18日的预测输入 state：data/processed/state_20250418.csv")