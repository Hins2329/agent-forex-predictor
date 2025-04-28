import pandas as pd

fx_path = "data/processed/usd_cny_minute_twelvedata.csv"
macro_path = "data/processed/macro_data.csv"
news_path = "data/processed/news_tone.csv"
output_path = "data/processed/state_minute.csv"

# loading
fx = pd.read_csv(fx_path, parse_dates=["datetime"])
macro = pd.read_csv(macro_path, parse_dates=["date"])
news = pd.read_csv(news_path, parse_dates=["datetime"])

# rating
fx = fx[["datetime", "close"]].rename(columns={"close": "exchange_rate"})
fx = fx.sort_values("datetime").reset_index(drop=True)

# marco
macro_daily = macro.set_index("date").resample("D").ffill().reset_index()
macro_daily = macro_daily.rename(columns={"date": "datetime"})
macro_daily = macro_daily.sort_values("datetime")

macro_minute = pd.merge_asof(fx, macro_daily, on="datetime")

# News sentiment: aggregate daily and then merge_asof (forward match)
news["date"] = news["datetime"].dt.date
news_daily = news.groupby("date")["tone"].mean().reset_index()
news_daily["datetime"] = pd.to_datetime(news_daily["date"])
news_daily = news_daily.sort_values("datetime")[["datetime", "tone"]]

state = pd.merge_asof(macro_minute, news_daily, on="datetime")

state = state.dropna().reset_index(drop=True)
state.to_csv(output_path, index=False)

print(f"✅ state_minute has been saved to: {output_path}")
print(state.tail())