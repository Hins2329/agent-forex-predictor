import torch
import pandas as pd
from agent.dqn_agent import DQNAgent
from env.forex_env import ForexEnv

# === 路径配置 ===
DATA_PATH = "/Users/junxuanzhang/Desktop/agent/data/processed/train_state.csv"
MODEL_PATH = "/Users/junxuanzhang/Desktop/agent/logs/dqn_model.pth"
LOG_PATH = "/Users/junxuanzhang/Desktop/agent/logs/training_log.csv"
PLOT_PATH = "/Users/junxuanzhang/Desktop/agent/logs/training_plot.png"
DEVICE = "cpu"

# === 环境与模型初始化 ===
env = ForexEnv(DATA_PATH, fee=0.0015)
state_dim = 6  # CPI, InterestRate, Unemployment, tone, exchange_rate, holding
action_dim = 3  # Hold, Buy, Sell
agent = DQNAgent(state_dim, action_dim, device=DEVICE)

# === 训练过程 ===
EPISODES = 300
TARGET_UPDATE_FREQ = 10

for episode in range(EPISODES):
    state = env.reset()  # 重置环境
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)  # 根据当前状态选择动作
        next_state, reward, done, profit = env.step(action)  # 获取四个值，注意增加了 profit
        agent.remember(state, action, reward, next_state, done)  # 记住当前状态、动作、奖励
        agent.learn()  # 学习更新 Q 网络
        state = next_state
        total_reward += reward  # 累计奖励

    # 每 N 轮同步目标网络
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target()
    # epsilon 衰减
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    print(f"✅ Episode {episode+1}/{EPISODES} - Total reward: {total_reward:.4f} - Epsilon: {agent.epsilon:.4f}")

# === 保存训练好的模型 ===
torch.save(agent.q_net.state_dict(), MODEL_PATH)
print("🎉 训练完成！模型已保存")