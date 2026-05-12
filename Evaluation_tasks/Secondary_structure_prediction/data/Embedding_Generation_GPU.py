import os
import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
import d2l.torch


def load_model_with_weights(base_dir,embedding_using_self_weight, selected_model, devices):
    # tune_model_0.5  tune_model_0.5_clans  tune_model_0.9  tune_model_0.9_clans  tune_model_all  tune_model_combine

    if embedding_using_self_weight is True and selected_model is not None:
        model_name = selected_model.split('_')[0]
        clus_coef = selected_model.split('_')[1]
        if selected_model[:2] == 'PF':
            model_path = f'{base_dir}/tune_model_{clus_coef}/{model_name}/{selected_model}.pth'
        elif selected_model[:2] == 'CL':
            model_path = f'{base_dir}/tune_model_{clus_coef}_clans/{model_name}/{selected_model}.pth'
        elif selected_model[:7] == 'combine':
            model_path = f'{base_dir}/tune_model_combine/{selected_model}/{selected_model}.pth'
        assert os.path.exists(model_path) is True
    else:
        model_path = None

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    model.eval()
    model = model.to(devices[0])
    batch_converter = alphabet.get_batch_converter()
    model_layer = model.num_layers  # Using .num_layers is more general
    
    return model, alphabet, batch_converter, model_layer

def create_embedding(df, base_dir, embedding_using_self_weight, devices, emb_type="per_prot", max_len=2048):
    # Filter data, remove sequences exceeding length limit
    original_count = len(df)
    df_filtered = df[df['sequence'].str.len() <= max_len].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    filtered_count = len(df_filtered)
    print(f"Original sequence count: {original_count}. After keeping sequences with length <= {max_len}, remaining: {filtered_count}. Discarded {original_count - filtered_count} sequences.")
    
    # Before grouping, replace NaN values with a special placeholder to avoid compatibility issues between tqdm and groupby.
    # This won't change the logic because all NaNs will still be grouped together.
    df_filtered['selected_model'].fillna('__NaN_GROUP__', inplace=True)

    # ==================== PER_PROT Mode ====================
    if emb_type == "per_prot":
        all_embeddings = [None] * len(df_filtered)
        for selected_model_key, group in tqdm(df_filtered.groupby("selected_model"), desc="[Prot] Processing models"):
            model_to_load = None if selected_model_key == '__NaN_GROUP__' else selected_model_key
            model, alphabet, batch_converter, model_layer = load_model_with_weights(base_dir, embedding_using_self_weight, model_to_load, devices)

            for idx, row in group.iterrows():  # idx is the reset index of df_filtered
                sequence = row["sequence"]
                targt_label, targt_str, targt_tokens = batch_converter([(f"seq_{idx}", sequence)])
                batch_tokens = targt_tokens.to(devices[0])

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[model_layer], return_contacts=False)
                    token_representations = results["representations"][model_layer]
                    embedding = token_representations[0, 1:len(sequence) + 1].mean(0).cpu().numpy()
                    all_embeddings[idx] = embedding

            del model
            torch.cuda.empty_cache()

        # Convert results to DataFrame
        df_emb = pd.DataFrame(all_embeddings)
        df_emb.columns = [f'emb_{k}' for k in range(df_emb.shape[1])]
        
        # Add labels and sequence info from filtered df_filtered, since order is guaranteed, direct assignment is possible
        # Note: Need to get from original df_filtered columns, not the replaced ones
        df_filtered['selected_model'].replace('__NaN_GROUP__', np.nan, inplace=True)  # Optional: restore NaN if needed later
        df_emb['sequence'] = df_filtered['sequence']
        df_emb['label'] = df_filtered['label']

    # ==================== PER_RES Mode ====================
    elif emb_type == "per_res":
        # === Add label validation logic ===
        def process_label(label, seq):
            if isinstance(label, list):
                if len(label) == len(seq): return label
                else: raise ValueError(f"label is a list with length {len(label)}, which does not match sequence length {len(seq)}.")
            elif isinstance(label, (int, str)):
                if len(str(label)) == len(seq): return [int(c) for c in str(label)]
                elif len(str(label)) == 1: return [label] * len(seq)
                else: raise ValueError(f"label is str/int with length {len(str(label))}, which does not match sequence length {len(seq)}.")
            else: raise TypeError(f"label type is {type(label)}, must be list, str, or int, and length must equal sequence.")
        df_filtered["label"] = df_filtered.apply(lambda row: process_label(row["label"], row["sequence"]), axis=1)

        # === mask check ===
        if "mask" in df_filtered.columns:
            def check_mask(mask, seq):
                if not isinstance(mask, str): raise TypeError(f"mask type is {type(mask)}, must be a string")
                if len(mask) != len(seq): raise ValueError(f"mask length is {len(mask)}, but sequence length is {len(seq)}, mismatch.")
                return mask
            df_filtered["mask"] = df_filtered.apply(lambda row: check_mask(row["mask"], row["sequence"]), axis=1)

        all_residue_embeddings = [None] * len(df_filtered)
        for selected_model_key, group in tqdm(df_filtered.groupby("selected_model"), desc="[Res] Processing models"):
            model_to_load = None if selected_model_key == '__NaN_GROUP__' else selected_model_key
            model, alphabet, batch_converter, model_layer = load_model_with_weights(base_dir, embedding_using_self_weight, model_to_load, devices)

            for idx, row in group.iterrows():
                sequence = row["sequence"]
                targt_label, targt_str, targt_tokens = batch_converter([(f"seq_{idx}", sequence)])
                batch_tokens = targt_tokens.to(devices[0])

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[model_layer], return_contacts=False)
                    token_representations = results["representations"][model_layer]
                    residue_embeddings = token_representations[0, 1:len(sequence) + 1].cpu().numpy()
                    all_residue_embeddings[idx] = residue_embeddings
            
            del model
            torch.cuda.empty_cache()

        final_embeddings_array = np.concatenate(all_residue_embeddings)
        df_emb = pd.DataFrame(final_embeddings_array)
        df_emb.columns = [f'emb_{k}' for k in range(df_emb.shape[1])]

        # --- Generate metadata (keep unchanged) ---
        df_filtered["pos_len"] = df_filtered['sequence'].str.len()
        df_filtered["label_list"] = df_filtered["label"] 
        df_filtered["idx_list"] = df_filtered.apply(lambda row: [row.name] * row["pos_len"], axis=1)
        df_filtered["pos_list"] = df_filtered.apply(lambda row: list(range(1, row["pos_len"] + 1)), axis=1)

        all_labels = np.concatenate(df_filtered['label_list'].values)
        all_idxs = np.concatenate(df_filtered['idx_list'].values)
        all_poss = np.concatenate(df_filtered['pos_list'].values)
        all_seqs_str = "".join(df_filtered['sequence'].tolist())

        df_emb['protein_idx'] = all_idxs
        df_emb['position'] = all_poss
        df_emb['residue'] = list(all_seqs_str)
        df_emb['label'] = all_labels

        if "mask" in df_filtered.columns:
            masks_str = "".join(df_filtered['mask'].tolist())
            df_emb["mask"] = list(masks_str)

    else:
        print("ERROR embedding type: 'per_prot' or 'per_res'")
        return None

    return df_emb


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    np.random.seed(42)

    input_dir = "./SecStr/"
    output_dir = "./esm2_all/"
    csv_path = 'data_with_model.csv'
    base_dir = '/home/zxc/human_proteome/esm2-weight'  # read me -> change the base path
    embedding_using_self_weight = False  # read me -> Use fine-tuned model (True) or pre-trained model (False)
    keep_empty_model = True  # Keep (True) or discard (False) empty model data
    emb_type = 'per_res'
    max_len = 2048

    # Read and prepare model mapping relationship
    model_map_df = pd.read_csv(csv_path)
    model_map_df = model_map_df[['sequence', 'selected_model']].drop_duplicates(subset=['sequence']).reset_index(drop=True)

    devices = d2l.torch.try_all_gpus()

    # Loop through train, valid, test files
    for data_split in ["test"]:
        input_pkl_path = os.path.join(input_dir, f"{data_split}.pkl")
        output_pkl_path = os.path.join(output_dir, f"{data_split}_embedding.pkl")
        
        df = pd.read_pickle(input_pkl_path)
        df = df.iloc[:,:3]
        df.columns = ["sequence", "label", "mask"]
        df["sequence"] = df["sequence"].str.replace('|'.join(["O","B","U","Z","J"]), "X", regex=True)

        # Merge to get 'selected_model' information
        df_merged = pd.merge(df, model_map_df, on='sequence', how='left')

        # Decide whether to drop data with empty 'selected_model' based on configuration
        if not keep_empty_model:
            initial_count = len(df_merged)
            df_merged.dropna(subset=['selected_model'], inplace=True)
            print(f"Configured to not keep empty model data. Discarded {initial_count - len(df_merged)} records.")

        # Call core function to generate embeddings
        emb = create_embedding(df_merged,base_dir,embedding_using_self_weight,devices,emb_type,max_len)
        emb.to_pickle(output_pkl_path)
