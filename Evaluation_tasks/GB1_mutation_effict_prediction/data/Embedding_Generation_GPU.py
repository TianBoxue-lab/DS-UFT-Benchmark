import os
import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import d2l.torch


def create_embedding(df, self_weight, devices, emb_type="per_prot"):

    # model, alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint_path)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    if self_weight is not None:
        model.load_state_dict(torch.load(self_weight))

    model.eval()
    model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    model_layer = 33

    emb = []

    if emb_type == "per_prot":
        for i in tqdm(range(len(df))):
            sequence = df["sequence"].loc[i]
            targt_label, targt_str, targt_tokens = batch_converter([(f"seq{i}", sequence)])
            batch_tokens = targt_tokens.cuda()
            batch_tokens = targt_tokens.to(devices[0])

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[model_layer], return_contacts=False)
                token_representations = results["representations"][model_layer]
                embedding = token_representations[0, 1:len(sequence)+1].mean(0).cpu().numpy()
                emb.append(embedding)

        df_emb = pd.DataFrame(np.concatenate(emb).reshape(len(emb), -1))
        df_emb.reset_index(drop=True, inplace=True)
        df_emb["sequence"] = df["sequence"]
        df_emb["label"] = df["label"]

    elif emb_type == "per_res":
        for i in tqdm(range(len(df))):
            sequence = df["sequence"].loc[i]
            targt_label, targt_str, targt_tokens = batch_converter([(f"seq{i}", sequence)])
            batch_tokens = targt_tokens.cuda()
            batch_tokens = targt_tokens.to(devices[0])

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[model_layer], return_contacts=False)
                token_representations = results["representations"][model_layer]
                residue_embeddings = token_representations[0, 1:len(sequence)+1].cpu().numpy()
                emb.append(residue_embeddings)

        df_emb = pd.DataFrame(np.concatenate(emb))
        df_emb.reset_index(drop=True, inplace=True)

        df["pos"] = df['sequence'].str.len()
        df["idx"] = df.index
        df["idx"] = df.apply(lambda x: np.array(x["pos"] * [x["idx"]]), axis=1)
        df["pos"] = df['pos'].apply(lambda x: np.array(range(1, x+1)))

        idxs = np.concatenate(df['idx'].values)
        poss = np.concatenate(df['pos'].values)
        df_emb["seq_idx_pos"] = [f"{aa}_{bb}" for aa, bb in zip(idxs, poss)]

        seqs = df['sequence'].str.cat()
        df_emb["residue"] = [aa for aa in seqs]

        labels = np.concatenate(df['label'].values)
        df_emb["label"] = [l for l in labels]

        if "mask" in df.columns:
            masks = df['mask'].str.cat()
            df_emb["mask"] = [m for m in masks]
    else:
        print("ERROR embedding type: 'per_prot' or 'per_res'")
        return None

    # clean
    del model
    torch.cuda.empty_cache()

    return df_emb

if __name__ == '__main__':
    self_weight = None
    # self_weight = '/home/user/data2/human_proteome/tasks/GB1/data/CL0041/CL0041_0.9.pth' # PF01335 CL0041

    for data in ["test", "valid", "train"]:
        path = f"./GB1/{data}.pkl"

        # dataframe with sequence and label
        df = pd.read_pickle(path).iloc[:,0:2]
        df.columns = ["sequence", "label"]

        devices = d2l.torch.try_all_gpus()

        # replace non common AAs
        df["sequence"]=df["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)
        emb = create_embedding(df, self_weight, devices, emb_type="per_prot")
        emb.to_pickle(f"./{data}_embedding.pkl")


