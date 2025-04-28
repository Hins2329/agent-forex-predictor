import pandas as pd

class ForexEnv:
    def __init__(self, data_path, fee=0.015):
        self.data = pd.read_csv(data_path, parse_dates=["datetime"])
        self.fee = fee
        self.reset()

    def reset(self):
        self.index = 0
        self.holding = 1  # 0: no holding, 1: holding
        self.entry_price = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done or self.index >= len(self.data) - 1:
            self.done = True
            return self._get_state(), 0, True, 0  # profit=0

        current_price = self.data.loc[self.index, "exchange_rate"]
        reward = 0
        profit = 0

        if action == 1:  # Buy
            if self.holding == 0:
                self.entry_price = current_price + self.fee  # entry price + fee
                self.holding = 1
            else:
                reward = -0.01  # rebuying

        elif action == 2:  # Sell
            if self.holding == 1:
                exit_price = current_price - self.fee  # sell - fee
                profit = exit_price - self.entry_price
                reward = profit * 100
                self.holding = 0
                self.entry_price = 0
            else:
                reward = -0.01  # selling when no holding


        elif action == 0:  # Hold
            reward = -0.0009
            profit = 0

        self.index += 1
        done = self.index >= len(self.data) - 1
        return self._get_state(), reward, done, profit


    def _get_state(self):
        row = self.data.iloc[self.index]
        return [
            row["CPI"],
            row["InterestRate"],
            row["Unemployment"],
            row["tone"],
            row["exchange_rate"],
            self.holding
        ]

    def render(self):
        print(f"Step {self.index}, Price={self.data.loc[self.index, 'exchange_rate']:.4f}, Holding={self.holding}")