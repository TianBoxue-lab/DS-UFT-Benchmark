import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import d2l.torch
from tqdm import tqdm
import time
import os
import torch
import esm
from torch import nn
import pandas as pd
from auxiliary import get_pfam_and_model_combination, gen_seq_list
import gc

def batch_list(test_list, x):
    # Split the list into a new list with groups of size x
    return [test_list[i: i + x] for i in range(0, len(test_list), x)]


def mask_position(batch_labels):
    positions = []
    for each in batch_labels:
        positions.append(int(each.split('_')[1]))
    return positions


def score_seq_batch(batch_labels, logits, targt_tokens):
    positions = mask_position(batch_labels)
    batch_ids = torch.arange(0, len(positions))
    logits_slice = logits[batch_ids, positions]
    targt_tokens_slice = targt_tokens[:, positions]  # positions are indices, i+1 due to start and end tokens
    loss = nn.CrossEntropyLoss(reduction='sum')
    ll_slice = -loss(logits_slice, targt_tokens_slice[0])  # Log likelihood, shape is [1, x]
    # print(ll_slice)
    return ll_slice


def prepare_perplexity_data(pfam_human, human_protein, model_name, pfam):
    # Generate data prepared for perplexity calculation
    ids_list, seq_list = gen_seq_list(pfam_human, human_protein, pfam)
    seq_num = len(ids_list)
    perp_data = dict()
    for name, seq in zip(ids_list, seq_list):
        len_seq = len(seq)
        data = []
        for i in range(len_seq):
            data.append((name + "_" + str(i + 1), seq[:i] + '<mask>' + seq[i + 1:]))  # Use i+1 instead of index in naming because start and end tokens are added to the input sequence
        perp_data[(name, seq)] = data
    return perp_data


def calculate_perplexity_for_pfam(model_name, pfam, perp_data, self_weight, devices, output_file_name):

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if self_weight is not None:
        model.load_state_dict(torch.load(self_weight))

    model.to(devices[0])
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    perplexity = dict()
    t = time.time()
    for name, seq in perp_data.keys():
        print('start to process sequence ' + name)
        targt_label, targt_str, targt_tokens = batch_converter([(name, seq)]) # original sequence
        seq_length = len(seq)
        if seq_length > 1024:
            # perplexity[name] = {'Pfam_id': model_name, 'Uniprot_id': name, "perplexity": None, "model_path": self_weight}
            print(f'ERROR: {name} more than 1024')

        if seq_length <= 1024:
            all_ll = 0
            for mask_data in batch_list(perp_data[(name, seq)], 8):
                # If CUDA out of memory, change batch size from 8 to a smaller number
                batch_labels, batch_strs, batch_tokens = batch_converter(mask_data)  # Masked sequence
                batch_tokens = batch_tokens.to(devices[0])
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                logits = results["logits"].to('cpu')
                all_ll = all_ll + score_seq_batch(batch_labels, logits, targt_tokens)

            perplexity[name] = {'Pfam_id': model_name, 'Uniprot_id': name, "perplexity": np.exp(-np.array(all_ll) / seq_length), "model_path": self_weight}
            del batch_tokens, batch_labels, batch_strs

        # clean    
        del targt_tokens, targt_str, targt_label
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f'cost:{time.time() - t:.4f}s')

    # ===== save metrics to CSV =====
    metrics_df = pd.DataFrame.from_dict(perplexity, orient="index")
    if os.path.exists(output_file_name):
        metrics_df.to_csv(output_file_name, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(output_file_name, mode='w', header=True, index=False)

    return perplexity


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    pfam_human = pd.read_csv('info_files/pfam_sequence_num.csv')[['pfam_id', 'human_uniprot_id']]  # Statistics of pfams in human proteome
    pfam_human.set_index("pfam_id", inplace=True, drop=True)

    human_protein = pd.read_csv('info_files/human_proteome_info.csv')[['uniprot_id', 'protein_sequence']]  # Information of human proteome
    human_protein.set_index("uniprot_id", inplace=True, drop=True)

    base_dir = '/home/zxc/human_proteome/esm2-weight'  # read me -> change the base path

    devices = d2l.torch.try_all_gpus()
    predict_using_self_weight = True  # read me -> Use fine-tuned model (True) or pre-trained model (False)

    models_list_file = 'info_files/all_pfam.txt'  # read me -> change file name, all_pfam all_clans all_combine
    output_file_name = "perplexity_pfam-fine-tune.csv" 
    pfam_and_model = get_pfam_and_model_combination(models_list_file)

    # for (pfam, model) in pfam_and_model.items():
    items_list = list(pfam_and_model.items())
    for (model_name, model) in items_list:
        if predict_using_self_weight is True:
            model_path = pjoin(base_dir, model)
            assert os.path.exists(model_path) is True
        else:
            model_path = None

        pfam = model_name.split('_')[0]
        perp_data = prepare_perplexity_data(pfam_human, human_protein, model_name, pfam)
        perplexity = calculate_perplexity_for_pfam(model_name, pfam, perp_data, model_path, devices, output_file_name)

