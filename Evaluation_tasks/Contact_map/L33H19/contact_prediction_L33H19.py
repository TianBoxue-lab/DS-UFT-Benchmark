import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from typing import Tuple
from scipy.spatial.distance import squareform, pdist, cdist
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import biotite.structure as bs
from biotite.structure.io.pdbx import PDBxFile, get_structure
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import esm
from pathlib import Path
from tqdm import tqdm
import d2l.torch
from auxiliary_zxc import get_pfam_and_model_combination
from os.path import join as pjoin
import time
import ast
import gc

# from esm.inverse_folding.contact import contacts_from_pdb
# from Bio.PDB import PDBParser


def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


def contacts_from_pdb(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:
    mask = ~structure.hetero
    if chain is not None:
        mask &= structure.chain_id == chain

    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]

    # Define contact map. Why not read CB positions directly? Instead, calculate CB positions from N, CA, C. Is it because CB might not be resolved?
    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143) 
    dist = squareform(pdist(Cbeta))
    
    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts


def expand_contacts(contacts_partial: np.ndarray, mask: np.ndarray, fill_value: int = -1):
    """
    Expand N×N contact map to M×M contact map (M is the original sequence length),
    fill missing positions with fill_value (e.g., -1)
    """
    M = len(mask)
    full_contacts = np.full((M, M), fill_value, dtype=np.int64)
    idx = np.where(mask)[0]  # Positions of residues with structure
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            full_contacts[ii, jj] = contacts_partial[i, j]
    return full_contacts


def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
        ("long", 24, None),
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics


"""Adapted from: https://github.com/rmrao/evo/blob/main/evo/visualize.py"""
def plot_contacts_and_predictions(
    predictions: Union[torch.Tensor, np.ndarray],
    contacts: Union[torch.Tensor, np.ndarray],
    ax: Optional[mpl.axes.Axes] = None,
    # artists: Optional[ContactAndPredictionArtists] = None,
    cmap: str = "Blues",
    ms: float = 1,
    title: Union[bool, str, Callable[[float], str]] = True,
    animated: bool = False,
) -> None:

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(contacts, torch.Tensor):
        contacts = contacts.detach().cpu().numpy()
    if ax is None:
        ax = plt.gca()

    seqlen = contacts.shape[0]
    relative_distance = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
    bottom_mask = relative_distance < 0
    masked_image = np.ma.masked_where(bottom_mask, predictions)
    invalid_mask = np.abs(np.add.outer(np.arange(seqlen), -np.arange(seqlen))) < 6
    predictions = predictions.copy()
    predictions[invalid_mask] = float("-inf")

    topl_val = np.sort(predictions.reshape(-1))[-seqlen]
    pred_contacts = predictions >= topl_val
    true_positives = contacts & pred_contacts & ~bottom_mask
    false_positives = ~contacts & pred_contacts & ~bottom_mask
    other_contacts = contacts & ~pred_contacts & ~bottom_mask

    if isinstance(title, str):
        title_text: Optional[str] = title
    elif title:
        long_range_pl = compute_precisions(predictions, contacts, minsep=24)[
            "P@L"
        ].item()
        if callable(title):
            title_text = title(long_range_pl)
        else:
            title_text = f"Long Range P@L: {100 * long_range_pl:0.1f}"
    else:
        title_text = None

    img = ax.imshow(masked_image, cmap=cmap, animated=animated)
    oc = ax.plot(*np.where(other_contacts), "o", c="grey", ms=ms)[0]
    fn = ax.plot(*np.where(false_positives), "o", c="r", ms=ms)[0]
    tp = ax.plot(*np.where(true_positives), "o", c="b", ms=ms)[0]
    ti = ax.set_title(title_text) if title_text is not None else None
    # artists = ContactAndPredictionArtists(img, oc, fn, tp, ti)

    ax.axis("square")
    ax.set_xlim([0, seqlen])
    ax.set_ylim([0, seqlen])


def gen_seq_list(pfam_human, human_protein_st, pfam):
    # Generate seqs based on the read in files
    ids_list = ast.literal_eval(pfam_human.loc[pfam, 'human_uniprot_id']) 

    new_id_list = []
    pdb_chain_list = []
    seq_st_full_list = []
    seq_st_list = []
    seq_st_align_list = []

    for each in ids_list:
        if each in human_protein_st.index:
            pdb_chain = human_protein_st.loc[each, 'select_pdb_chain']
            seq_st_full = human_protein_st.loc[each, 'seq_st_full']
            seq_st = human_protein_st.loc[each, 'seq_st']
            seq_st_align = human_protein_st.loc[each, 'seq_st_align']

            new_id_list.append(each)
            pdb_chain_list.append(pdb_chain)
            seq_st_full_list.append(seq_st_full)
            seq_st_list.append(seq_st)
            seq_st_align_list.append(seq_st_align)

    return new_id_list, pdb_chain_list, seq_st_full_list, seq_st_list, seq_st_align_list


def prepare_conc_data(pfam_human, human_protein_st, model_name, pfam):
    # Generate data prepared for contact calculation
    ids_list, pdb_chain_list, seq_st_full_list, seq_st_list, seq_st_align_list = gen_seq_list(pfam_human, human_protein_st, pfam)

    seq_num = len(ids_list)
    conc_data = dict()
    for name, pdb_chain, seq_st_full,seq_st,seq_st_align in zip(ids_list, pdb_chain_list, seq_st_full_list,seq_st_list,seq_st_align_list):
        conc_data[name] = (pdb_chain, seq_st_full,seq_st,seq_st_align)

    return conc_data


def save_contact_map_as_image(matrix, filepath, title=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='viridis', origin='lower')
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    return True


def calculate_contact_for_pfam_and_save(model_name, pfam, concmap_data, cif_dir, save_dir, self_weight, devices, output_file_name):

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    if self_weight is not None:
        model.load_state_dict(torch.load(self_weight))
        sub_dir = 'tuned'
    else:
        sub_dir = 'no-tuned'
    save_path = pjoin(save_dir, model_name, sub_dir)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    model.to(devices[0])
    model.eval()  # disables dropout for deterministic results
    batch_converter = alphabet.get_batch_converter()

    contact_map = dict()
    t = time.time()

    if concmap_data:
        for uniprot, seq_info in concmap_data.items():
            print(f'start to process sequence {model_name}-{uniprot}')

            pdb_chain, seq_st_full, seq_st, seq_st_align = seq_info
            pdb_id, chain_id = pdb_chain.split("_")
            if 'nan' in chain_id:
                chain_id = 'NA'

            if len(seq_st) >= 50 and len(seq_st_full) <= 1024:
                # Get true contact map
                cif_path = pjoin(cif_dir, f"{pdb_id.lower()}.cif")
                if not os.path.exists(cif_path):
                    print(f"[Skip] Structure file not found: {cif_path}")
                    continue

                try:
                    structure = get_structure(PDBxFile.read(cif_path))[0]
                    real_contact_map = contacts_from_pdb(structure, chain=chain_id)
                except Exception as e:
                    print(f"[Skip] Structure parsing failed: {pdb_chain} - {e}")
                    continue

                if len(seq_st) != real_contact_map.shape[0]:
                    print(f"⚠️ seq_st sequence length does not match contact map size! {uniprot}")

                # ESM2 predicted contact map
                with torch.no_grad():
                    targt_label, targt_str, targt_tokens = batch_converter([(uniprot, seq_st_full)])
                    batch_tokens = targt_tokens.to(devices[0])
                    pred_contact_map = model.predict_contacts(batch_tokens)[0].cpu().numpy()
                #     results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                # contact_map = results["contacts"].to('cpu')[0]

                # Mask to filter residues with missing structure
                valid_mask = np.array([aa != "-" for aa in seq_st_align])
                pred_contact_map_cut = pred_contact_map[valid_mask][:, valid_mask]
                # full_contacts = expand_contacts(contact_map, valid_mask) #从缺失扩展到完整
                if len(seq_st) != pred_contact_map_cut.shape[0]:
                    print(f"⚠️ seq_st sequence length does not match truncated contact map size! {uniprot}")

                # Apply mask for missing residues
                valid_mask = np.array([aa != "-" for aa in seq_st_align])
                pred_contact_map_cut = pred_contact_map[valid_mask][:, valid_mask]

                # ===== Evaluation =====
                metrics = {'Model_name': model_name, "uniprot": uniprot, "pdb_chain": pdb_chain}
                try:
                    metrics.update(evaluate_prediction(torch.tensor(pred_contact_map_cut), torch.tensor(real_contact_map)))
                except Exception as e:
                    print(f"⚠️ Evaluation failed: {uniprot} - {e}")
                    continue

                # ===== Save metrics to CSV =====
                metrics_df = pd.DataFrame([metrics]) 
                if os.path.exists(output_file_name):
                    metrics_df.to_csv(output_file_name, mode='a', header=False, index=False)
                else:
                    metrics_df.to_csv(output_file_name, mode='w', header=True, index=False)

                # ===== Save images and matrices =====
                np.save(pjoin(save_path, f"{uniprot}_{pdb_chain}_true_contact.npy"), real_contact_map)
                save_contact_map_as_image(real_contact_map,
                    pjoin(save_path, f"{uniprot}_{pdb_chain}_true_contact.png"),
                    title="True Contact Map")

                np.save(pjoin(save_path, f"{uniprot}_{pdb_chain}_pred_full.npy"), pred_contact_map)
                save_contact_map_as_image(pred_contact_map,
                    pjoin(save_path, f"{uniprot}_{pdb_chain}_pred_full.png"),
                    title="ESM Predicted Full Contact")

                np.save(pjoin(save_path, f"{uniprot}_{pdb_chain}_pred_cut.npy"), pred_contact_map_cut)
                save_contact_map_as_image(pred_contact_map_cut,
                    pjoin(save_path, f"{uniprot}_{pdb_chain}_pred_cut.png"),
                    title="ESM Predicted (Filtered)")

                # ===== Plot (with evaluation score) =====
                fig, ax = plt.subplots(figsize=(6, 6))
                plot_contacts_and_predictions(pred_contact_map_cut, real_contact_map, ax=ax,
                    title=lambda x: f"{uniprot} ({pdb_chain})\nLong Range P@L: {100 * x:.1f}")
                plt.tight_layout()
                plt.savefig(pjoin(save_path, f"{uniprot}_{pdb_chain}_compare.png"))
                plt.close()

                # Clean up intermediate variables to prevent accumulation
                del batch_tokens, targt_tokens, targt_str, targt_label
                del real_contact_map, pred_contact_map, pred_contact_map_cut, metrics_df, fig, ax
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            print(f"Completed {model_name}-{uniprot}, time cost: {time.time() - t:.2f}s")


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    pfam_human = pd.read_csv('info_files/pfam_sequence_num.csv')[['pfam_id', 'human_uniprot_id']]
    pfam_human.set_index("pfam_id", inplace=True, drop=True)

    human_protein = pd.read_csv('info_files/human_proteome_info.csv')[['uniprot_id', 'protein_sequence']]
    human_protein.set_index("uniprot_id", inplace=True, drop=True)

    base_dir = '/home/zxc/human_proteome/esm2-weight'  # read me -> change the base path

    devices = d2l.torch.try_all_gpus()
    predict_using_self_weight = False  # read me -> Use fine-tuned model (True) or pre-trained model (False)

    models_list_file = 'info_files/all_pfam.txt'  # read me -> change file name, all_pfam all_clans all_combine
    output_dir = "contact_map_pfam"
    output_file_name = "contact_map_pfam_ori.csv" 

    # Load structure files uniprot-pdb-seq
    cif_dir = "/home/zxc/human_proteome/PLM/cif"
    human_protein_st = pd.read_csv("human_proteome_info_with_select_pdb_chain_change_sequence.csv")
    human_protein_st = human_protein_st[~human_protein_st["select_pdb_chain"].isnull()].copy()
    human_protein_st.set_index("uniprot_id", inplace=True, drop=True)
    
    pfam_and_model = get_pfam_and_model_combination(models_list_file)
    # for (pfam, model) in pfam_and_model.items():
    
    # Only iterate through first 1000 elements
    items_list = list(pfam_and_model.items())
    for (model_name, model) in items_list:
        torch.cuda.empty_cache()
        if predict_using_self_weight is True:
            model_path = pjoin(base_dir, model)
            assert os.path.exists(model_path) is True
        else:
            model_path = None

        pfam = model_name.split('_')[0]
        concmap_data = prepare_conc_data(pfam_human, human_protein_st, model_name, pfam)
        calculate_contact_for_pfam_and_save(model_name, pfam, concmap_data, cif_dir, output_dir, model_path, devices, output_file_name)

