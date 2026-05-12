# convert_data.py
import pandas as pd
import numpy as np
import os

data_dir = "./data/"
files_to_convert = ['train', 'valid', 'test']

for f_type in files_to_convert:
    pkl_path = os.path.join(data_dir, f"{f_type}_embedding.pkl")
    print(f"Converting {pkl_path}...")

    df = pd.read_pickle(pkl_path)

    embedding_cols = [col for col in df.columns if col.startswith('emb_')]
    embeddings_data = df[embedding_cols].values.astype(np.float32)
    labels_data = df['label'].values.astype(np.int64)

    # 定义新的 .npy 文件路径
    emb_npy_path = os.path.join(data_dir, f"{f_type}_embeddings.npy")
    lbl_npy_path = os.path.join(data_dir, f"{f_type}_labels.npy")

    # 保存
    np.save(emb_npy_path, embeddings_data)
    np.save(lbl_npy_path, labels_data)

print("All files converted successfully!")
