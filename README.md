# USD/CNY Trading Agent: Maximize Profit with Macroeconomics, News Sentiment, and Exchange Rates

This project builds a trading agent that aims to maximize profit by trading USD and CNY.  
The agent leverages multi-source data including macroeconomic indicators, minute-level exchange rates, and news sentiment scores.

---

## 📦 Data Sources

| Source | Data | Description |
|:------|:----|:------------|
| **FRED** | Macro Data | Monthly CPI, Interest Rate, Unemployment |
| **TwelveData** | Exchange Rates | USD/CNY minute-level OHLCV data |
| **GDELT / NewsAPI** | News Sentiment | Daily aggregated news tones (positive/negative) |

---

## 📄 Dataset Details

- **Macro Data**
  - Columns: `date, CPI, InterestRate, Unemployment`
  - Frequency: Monthly → Resampled to Daily

- **Exchange Rates**
  - Columns: `datetime, open, high, low, close, volume`
  - Frequency: Minute-level

- **News Tone**
  - Columns: `date, tone, source_url`
  - Tone Score: Aggregated sentiment for economic news (positive > 0, negative < 0)

- **Final Training Data (`state_minute.csv`)**
  - Columns: `datetime, CPI, InterestRate, Unemployment, tone, exchange_rate`
  - Frequency: 1-minute, with forward-filled macro/news features.

---

## ⚙️ Agent Training & Testing

- **RL Algorithm**: DQN (Deep Q-Network)
- **Environment**: Custom `ForexEnv`
- **State Features**: Macro indicators + News tone + Exchange rate + Holding state
- **Action Space**:  
  - `0`: Hold  
  - `1`: Buy USD  
  - `2`: Sell USD
- **Reward**: Based on trading profit, considering transaction fees.
- **Goal**: Maximize cumulative profit over time.

---

## 🚧 Known Limitations

- After 300 episodes, the agent tends to be overly conservative (holding too long).
- Only a few trades executed; cumulative profit is still not optimal.
- Training reward remains negative, indicating **underfitting**.
- **TODOs**:
  - Increase number of training episodes (e.g., >1000).
  - Improve exploration strategies (adjust ε-decay settings).
  - Predict exchange rate changes instead of only deciding actions.
  - Apply more advanced algorithms (Dueling DQN, Double DQN, PPO, etc.).
- Since the training machine is a MacBook with an M1 chip, plans for further training with a larger number of episodes are currently on hold.
---

## 🧠 News Sentiment Enhancement (BERT Fine-Tuning)

- The original tone scores from APIs (GDELT/TwelveData) are **not ideal** for precise trading tasks.
- Therefore, a custom **economic news sentiment classifier** is being developed using **BERT**.
- A simple test on 10 economic text categories shows ~70% accuracy.
- **Future Plans**:
  - Expand training dataset for better domain-specific fine-tuning.
  - Improve news tone extraction quality to help the agent better predict market movements.

---

## 💡 Future Improvements

- Integrate a **custom BERT-based sentiment analyzer**.
- Predict future exchange rates alongside trading decisions.
- Build a **multi-modal agent** combining textual, numerical, and market data.
- Enable real-time adaptive trading against USD/CNY minute-level feeds.

---

# 📈 Final Goal

Create an intelligent agent that **understands macro trends, news sentiment, and market patterns**  
to **maximize USD/CNY trading profits** in a continuously changing environment.