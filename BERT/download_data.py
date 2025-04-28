# download_data.py
from datasets import load_dataset

def load_financial_data():
    dataset = load_dataset("financial_phrasebank", "sentences_allagree")
    return dataset

if __name__ == "__main__":
    data = load_financial_data()

    df = data['train'].to_pandas()

    df.to_csv("data/financial_phrasebank.csv", index=False)

    print("保存完成！")