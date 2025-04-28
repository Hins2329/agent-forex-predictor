import pandas as pd
import torch
import os
from agent.dqn_agent import QNetwork
from env.forex_env import ForexEnv
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/state_minute.csv"
MODEL_PATH = "logs/dqn_model.pth"
LOG_PATH = "logs/predict_profit_log.csv"
PLOT_PATH = "logs/predict_profit_curve.png"
DEVICE = "cpu"

env = ForexEnv(DATA_PATH)
env.holding = 1
env.entry_price = 7.22  # entry price

model = QNetwork(state_dim=6, action_dim=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === initial variable ===
state = env.reset()
done = False
log = []
cumulative_profit = 0
buy_price = 7.22

while not done:
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    dt = env.data.loc[env.index, "datetime"]
    price_before = env.data.loc[env.index, "exchange_rate"]
    holding_before = state[5]         # holding before action?
    entry_price_before = env.entry_price  # holding price

    next_state, reward, done, profit = env.step(action)
    cumulative_profit += profit


    # Actions
    if action == 1 and holding_before == 0:
        buy_price=env.data.loc[env.index, "exchange_rate"] + env.fee
        print(f"[BUY ] {dt.strftime('%Y-%m-%d %H:%M')} at {price_before + env.fee:.4f}")
    elif action == 2 and holding_before == 1:
        sell_price = env.data.loc[env.index, "exchange_rate"] - env.fee
        true_profit = sell_price - buy_price
        print(f"[SELL] {dt.strftime('%Y-%m-%d %H:%M')} at {sell_price:.4f} → Profit: {true_profit:.4f}")

    log.append({
        "datetime": dt,
        "exchange_rate": price_before,
        "CPI": state[0],
        "InterestRate": state[1],
        "Unemployment": state[2],
        "tone": state[3],
        "holding": int(state[5]),
        "action": action,
        "profit": profit,
        "cumulative_profit": cumulative_profit
    })

    state = next_state

# === Saving log ===
os.makedirs("logs", exist_ok=True)
df = pd.DataFrame(log)
df.to_csv(LOG_PATH, index=False)
print(f"✅ Logs have been saved to : {LOG_PATH}")

# === Graph ===
plt.figure(figsize=(14, 6))
plt.plot(df["datetime"], df["cumulative_profit"], label="Cumulative Profit", color="blue")
plt.title("DQN Agent Cumulative Profit (Jan–Apr)")
plt.xlabel("Datetime")
plt.ylabel("Cumulative Profit (RMB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📈 Graph has been saved to : {PLOT_PATH}")