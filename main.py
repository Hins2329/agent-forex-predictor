# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

def print_hi(name):
    print(f'Hi, {name}')


log_dir = "/logs"
MODEL_PATH = "/Users/junxuanzhang/Desktop/agent/logs/dqn_model.pth"


if os.path.exists(MODEL_PATH):
    print("exists")
else:
    print("not found")

if __name__ == '__main__':
    print_hi('PyCharm')

