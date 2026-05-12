import random
import multiprocessing
import torch
import esm
import os
import d2l.torch
import pandas as pd
from typing import List
from tqdm import tqdm
import math
from data_process import delete_data_out

alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

def mkdirs(dirs):
    """
    Function to make directories iteratively.
    Args:
        @arg dirs: a list or a string contains the path(s) to create.
    """

    if isinstance(dirs, list):
        for dir_path in dirs:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
    elif isinstance(dirs, str):
        if not os.path.isdir(dirs):
            os.makedirs(dirs)


def load_dataset_from_csv(csv_path: str,
                          csv_keys: List[str]):
    # prepare keys
    csv = pd.read_csv(csv_path)
    data_len = len(csv)

    # load all from csv
    d_strs = []
    for line in range(data_len):
        for key in csv_keys:
            d_strs.append(csv.loc[line][key])

    del csv
    return d_strs


def load_dataset_from_txt(txt_path: str):

    d_strs = []
    with open(txt_path, 'r') as infile:
        for line in tqdm(infile):
            data_line = line.strip("\n").split()[0]  # Remove leading/trailing newlines and split by space
            d_strs.append(data_line)

    return d_strs


def load_dataset_from_fasta(fasta_path: str,
                            g_mix):
    # Read sequences line by line
    # Input FASTA file, return sequences
    # g_mix, GaussianMixture whether to process sequence length with Gaussian Mixture Model
    d_strs = []

    with open(fasta_path, 'r') as infile:
        name, seq = '', ''

        while 1:
            line = infile.readline()
            line = line.strip('\n')  # Remove leading and trailing whitespace from the string
            if (line.startswith('>') or not line) and name:
                d_strs.append(seq)
            if line.startswith('>'):
                name = line[1:]
                seq = ''
            else:
                seq += line
            if not line:
                break

    return delete_data_out(d_strs, 2, g_mix)


class getDataset(torch.utils.data.Dataset):
    def __init__(self, seq_str_list, truncation_seq_length):

        self.truncation_seq_length = truncation_seq_length

        seq_tokens_list = self._tokenize(seq_str_list)

        (self.all_tokens_id, self.all_positions, self.all_labels) = self._preprocess(seq_tokens_list)
        print(f'read {len(seq_str_list)} examples')
        print(f'process to {len(self.all_tokens_id)} sequences')

    def _tokenize(self, seq_str_list):
        seq_tokens_list = [alphabet.tokenize(seq_str) for seq_str in seq_str_list]

        # Replace tokens not in vocabulary with '<unk>'
        for seq_tokens in seq_tokens_list:
            for token in seq_tokens:
                if token not in alphabet.all_toks:
                    seq_tokens[seq_tokens.index(token)] = '<unk>'

        # Random hard truncation
        if self.truncation_seq_length:
            true_trunc = self.truncation_seq_length - 2
            truncation_seq_tokens_list = []
            for seq_tokens in seq_tokens_list:
                if len(seq_tokens) <= true_trunc:
                    truncation_seq_tokens_list.append(seq_tokens)
                else:
                    num = math.floor(len(seq_tokens) / true_trunc)
                    remainder = len(seq_tokens) % true_trunc

                    for each in range(num):
                        R = random.randint(0 + each * true_trunc,
                                           remainder + each * true_trunc)
                        truncation_seq_tokens_list.append(seq_tokens[R: R + true_trunc])
            seq_tokens_list = truncation_seq_tokens_list

        # Add <cls> and <eos> tokens
        if alphabet.prepend_bos:
            seq_tokens_list = [['<cls>'] + seq_tokens for seq_tokens in seq_tokens_list]
        if alphabet.append_eos:
            seq_tokens_list = [seq_tokens + ['<eos>'] for seq_tokens in seq_tokens_list]

        return seq_tokens_list

    def _preprocess(self, seq_tokens_list):
        pool = multiprocessing.Pool(4)  # Use 4 processes
        out = pool.map(self._mp_worker, seq_tokens_list)

        all_tokens_id = [tokens_id for tokens_id, positions, labels in out]
        all_positions = [positions for tokens_id, positions, labels in out]
        all_labels = [labels for tokens_id, positions, labels in out]

        # Currently uncertain about the impact of data type on model training
        return all_tokens_id, all_positions, all_labels

    def _mp_worker(self, seq_tokens):
        return self._get_mlm_data_from_tokens(seq_tokens)

    # Use masked tokens to replace input token sequence as model input, represent them with vocab index IDs,
    # along with prediction position indices and prediction position label token vocab index IDs
    # Get masked input sequence, masked positions (position IDs to predict for each sequence),
    # and original tokens at prediction positions as labels
    def _get_mlm_data_from_tokens(self, tokens):
        candidate_pred_positions = []

        # tokens is a list of strings
        for i, token in enumerate(tokens):
            # Special tokens are not predicted in masked language model tasks
            if token in ('<cls>', '<eos>'):
                continue  # Do not predict special tokens '<cls>', '<eos>'
            else:
                candidate_pred_positions.append(i)

        # Predict 15% random tokens in masked language model task
        num_mlm_preds = max(1, round(len(tokens) * 0.15))  # Number of predictions per sequence
        mlm_input_tokens, positions_and_labels = self._replace_mlm_tokens(tokens, candidate_pred_positions,
                                                                          num_mlm_preds)

        positions_and_labels = sorted(positions_and_labels, key=lambda x: x[0])  # Sort by prediction position index
        positions = [v[0] for v in positions_and_labels]  # Get prediction position indices
        labels = [v[1] for v in positions_and_labels]  # Get tokens at prediction positions

        return [alphabet.get_idx(each) for each in mlm_input_tokens], positions, [alphabet.get_idx(each) for each in labels]

    # Randomly select some tokens in input sequence to replace with '<mask>' or other tokens
    def _replace_mlm_tokens(self, tokens, candidate_pred_positions, num_mlm_preds):
        # Create new token copy for masked language model input, which may contain replaced "<mask>" or random tokens
        mlm_input_tokens = [token for token in tokens]
        pred_positions_and_labels = []
        # Shuffle to get 15% random tokens for prediction in masked language model task
        random.shuffle(candidate_pred_positions)
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions_and_labels) >= num_mlm_preds:
                break
            mask_token = None
            # 80% probability: replace token with "<mask>"
            if random.random() < 0.8:
                mask_token = '<mask>'
            else:
                # 10% probability: keep token unchanged
                if random.random() < 0.5:
                    mask_token = tokens[mlm_pred_position]
                # 10% probability: replace with random token
                else:
                    mask_token = random.choice(alphabet.all_toks)
            mlm_input_tokens[mlm_pred_position] = mask_token
            pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
        return mlm_input_tokens, pred_positions_and_labels

    # This function enables batch computation using matrices (tensors), keeping tensor shapes consistent
    def _pad_bert_inputs(self, example):
        max_num_mlm_preds = round(self.max_len * 0.15)  # round() function rounds to nearest integer

        tokens_id, pred_positions, pred_labels = example

        tokens_id = tokens_id + [alphabet.padding_idx] * (self.max_len - len(tokens_id))

        # [0] is used for padding only, filtered out by 0 weights when calculating loss
        positions = pred_positions + [0] * (max_num_mlm_preds - len(pred_labels))

        # Padding token predictions will be filtered out in loss by multiplying with 0 weights
        weights = [1.0] * len(pred_labels) + [0.0] * (max_num_mlm_preds - len(pred_labels))

        labels = pred_labels + [0] * (max_num_mlm_preds - len(pred_labels))
        return tokens_id, positions, weights, labels

    def __getitem__(self, idx):
        return self.all_tokens_id[idx], self.all_positions[idx], self.all_labels[idx]

    def __len__(self):
        return len(self.all_tokens_id)


def collate_func(batch):
    """Adjust padding length to be adaptive"""
    all_tokens_id, all_labels, all_positions, all_weights = [], [], [], []

    max_len = max(len(tokens_id) for (tokens_id, positions, labels) in batch)
    max_num_mlm_preds = round(max_len * 0.15)  # round() function rounds to nearest integer

    for (tokens_id, positions, labels) in batch:
        all_tokens_id.append(tokens_id + [alphabet.padding_idx] * (max_len - len(tokens_id)))

        all_positions.append(positions + [0] * (max_num_mlm_preds - len(labels)))
        # [0] is used for padding only, filtered out by 0 weights when calculating loss

        all_labels.append(labels + [0] * (max_num_mlm_preds - len(labels)))

        # Padding token predictions will be filtered out in loss by multiplying with 0 weights
        all_weights.append([1.0] * len(labels) + [0.0] * (max_num_mlm_preds - len(labels)))

    return (torch.tensor(all_tokens_id, dtype=torch.int64),
            torch.tensor(all_positions, dtype=torch.int64),
            torch.tensor(all_weights, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.int64))


def load_data_affinity(batch_size, num_workers, truncation_seq_length: int = None):
    """Load affinity dataset"""
    filename = "affinity_prediction/all_data.csv"
    csv_keys = ['antibody_a', 'antibody_b']
    data_list = load_dataset_from_csv(filename, csv_keys)
    timer = d2l.torch.Timer()
    timer.start()
    data_set = getDataset(data_list, truncation_seq_length)
    timer.stop()
    print('Processing all data in', timer.sum(), 's')
    data_iter = torch.utils.data.DataLoader(data_set, batch_size, num_workers=num_workers,
                                            shuffle=True, collate_fn=collate_func)
    return data_iter


def load_data_alpaca(batch_size, num_workers, truncation_seq_length: int = None):
    """Load alpaca dataset"""
    filename = "alpaca_nano/processed_data.txt"
    data_list = load_dataset_from_txt(filename)
    timer = d2l.torch.Timer()
    timer.start()
    data_set = getDataset(data_list, truncation_seq_length)
    timer.stop()
    print('Processing all data in', timer.sum(), 's')
    data_iter = torch.utils.data.DataLoader(data_set, batch_size, num_workers=num_workers,
                                            shuffle=True, collate_fn=collate_func)
    return data_iter


def load_data_expression(batch_size, num_workers, truncation_seq_length: int = None):
    """Load expression dataset"""
    filename = "expression/raw_jingrui_filter.txt"
    data_list = load_dataset_from_txt(filename)
    timer = d2l.torch.Timer()
    timer.start()
    data_set = getDataset(data_list, truncation_seq_length)
    timer.stop()
    print('Processing all data in', timer.sum(), 's')
    data_iter = torch.utils.data.DataLoader(data_set, batch_size, num_workers=num_workers,
                                            shuffle=True, collate_fn=collate_func)
    return data_iter


def load_data_human(file_name, batch_size, num_workers, truncation_seq_length: int = None, g_mix=None):
    """Load pfam_human dataset"""
    data_list = load_dataset_from_fasta(file_name, g_mix)
    timer = d2l.torch.Timer()
    timer.start()
    data_set = getDataset(data_list, truncation_seq_length)
    timer.stop()
    print('Processing all data in', timer.sum(), 's')
    data_iter = torch.utils.data.DataLoader(data_set, batch_size, num_workers=num_workers,
                                            shuffle=True, collate_fn=collate_func)
    return data_iter, len(data_set.all_tokens_id)


if __name__ == "__main__":

    batch_size, num_workers, truncation_seq_length, g_mix  = 16, 16, 1024, None
    # data_iter = load_data_expression(batch_size, num_workers, truncation_seq_length)
    file_name = '/homes/Tianlab/weiming/pfam/PF16197/PF16197_0.5.fasta'
    data_iter, _ = load_data_human(file_name, batch_size, num_workers, truncation_seq_length, g_mix)
    # for (tokens_id_X, mlm_positions_X, mlm_weights_X, mlm_positions_Y) in data_iter:
    #     print(tokens_id_X.shape, mlm_positions_X.shape, mlm_weights_X.shape, mlm_positions_Y.shape)