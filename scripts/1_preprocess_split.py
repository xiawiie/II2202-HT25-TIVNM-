import sys
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config


def preprocess_and_split():
    print(f"Loading raw CSV from: {Config.RAW_LABEL_CSV}")
    if not os.path.exists(Config.RAW_LABEL_CSV):
        print("❌ Error: 找不到 CSV 文件，请检查 config.py 中的 RAW_LABEL_CSV 路径")
        return

    df = pd.read_csv(Config.RAW_LABEL_CSV)

    # 聚合数据
    df_grouped = df.groupby('patientId')['Target'].max().reset_index()

    print(f"Total patients: {len(df_grouped)}")
    print(f"Positive samples: {df_grouped['Target'].sum()}")

    # 划分数据集
    X = df_grouped
    y = df_grouped['Target']

    # 80% train+val, 20% test
    train_val, test = train_test_split(X, test_size=0.2, random_state=Config.SEED, stratify=y)
    # 10% val (0.125 of 80% = 10% total)
    train, val = train_test_split(train_val, test_size=0.125, random_state=Config.SEED, stratify=train_val['Target'])

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    train.to_csv(os.path.join(Config.PROCESSED_DIR, 'train.csv'), index=False)
    val.to_csv(os.path.join(Config.PROCESSED_DIR, 'val.csv'), index=False)
    test.to_csv(os.path.join(Config.PROCESSED_DIR, 'test.csv'), index=False)

    print(f"✅ Data processed and saved to {Config.PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_and_split()