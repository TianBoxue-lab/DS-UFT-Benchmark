import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import gc
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import esm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===========================
# 1. Basic Configuration and Utility Functions
# ===========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# --- New: Helper function to save only trainable parameters ---
def save_trainable_model(model, path):
    """
    Save only parameters with requires_grad=True, significantly reducing file size.
    """
    trainable_state_dict = {k: v for k, v in model.state_dict().items() 
                            if k in [n for n, p in model.named_parameters() if p.requires_grad]}
    torch.save(trainable_state_dict, path)

# ===========================
# 2. Data Processing Classes
# ===========================
class ProteinDataset(Dataset):
    def __init__(self, pkl_path):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Data file not found: {pkl_path}")
        df = pd.read_pickle(pkl_path).iloc[:, :2]
        df.columns = ["sequence", "label"]
        # Simple cleaning
        df["sequence"] = df["sequence"].str.replace(r"[OBUZJ]", "X", regex=True)
        self.seqs = df["sequence"].tolist()
        self.labels = df["label"].astype(float).tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

class ESMBatchConverter:
    def __init__(self, alphabet, max_len):
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.max_len = max_len

    def __call__(self, batch):
        seqs, labels = zip(*batch)
        raw_batch = [("", seq) for seq in seqs]
        _, _, tokens = self.batch_converter(raw_batch)
        
        # Truncate sequences that are too long
        if tokens.size(1) > self.max_len:
            tokens = tokens[:, :self.max_len]
            
        return tokens, torch.tensor(labels, dtype=torch.float32)

# ===========================
# 3. Model Definition
# ===========================
class ESMRegressionHead(nn.Module):
    def __init__(self, model_name, unfreeze_layers, hidden_dim, dropout, custom_weights_path=None):
        super().__init__()
        
        # Load pretrained ESM
        self.esm, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.repr_layer = self.esm.num_layers
        embed_dim = self.esm.embed_dim

        # If custom weights need to be loaded (full weights)
        if custom_weights_path and os.path.exists(custom_weights_path):
            print(f"  [Model] Loading CUSTOM weights from: {custom_weights_path}")
            checkpoint = torch.load(custom_weights_path, map_location="cpu")
            if "model" in checkpoint: state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint: state_dict = checkpoint["state_dict"]
            else: state_dict = checkpoint
            # Remove module. prefix
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.esm.load_state_dict(new_state_dict, strict=False)
        
        # 1. Freeze all parameters
        for param in self.esm.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze last N layers
        if unfreeze_layers > 0:
            for layer in self.esm.layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # LayerNorm also needs to be unfrozen
            for param in self.esm.emb_layer_norm_after.parameters():
                param.requires_grad = True

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tokens):
        out = self.esm(tokens, repr_layers=[self.repr_layer], return_contacts=False)
        token_repr = out["representations"][self.repr_layer]

        # Padding Mask
        padding_mask = (tokens != self.alphabet.padding_idx) & \
                       (tokens != self.alphabet.cls_idx) & \
                       (tokens != self.alphabet.eos_idx)
        padding_mask = padding_mask.unsqueeze(-1).float()
        
        # Mean Pooling
        masked_sum = (token_repr * padding_mask).sum(dim=1)
        seq_len = padding_mask.sum(dim=1)
        seq_emb = masked_sum / (seq_len + 1e-8)

        return self.head(seq_emb).squeeze(-1)

# ===========================
# 4. Training Core Logic
# ===========================

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    # Prevent nan errors
    if len(y_true) < 2: 
        return {"spearman": 0, "pearson": 0, "r2": 0, "mse": mse, "rmse": np.sqrt(mse), "mae": 0}
        
    return {
        "spearman": spearmanr(y_true, y_pred).correlation,
        "pearson": pearsonr(y_true, y_pred)[0],
        "r2": r2_score(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred)
    }

def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0
    # leave=False to avoid screen flooding
    for tokens, labels in tqdm(loader, desc="    Train", leave=False):
        tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast():
            preds = model(tokens)
            loss = F.mse_loss(preds, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for tokens, labels in loader:
        tokens = tokens.to(DEVICE)
        with autocast():
            out = model(tokens)
        preds.extend(out.float().cpu().numpy())
        trues.extend(labels.numpy())
    return calculate_metrics(np.array(trues), np.array(preds))

def run_experiment(config, run_name="exp", verbose=True):
    set_seed(config["seed"])
    
    # Result save path
    save_path = os.path.join("results", f"{run_name}_best.pth")
    
    # 1. Model initialization
    model = ESMRegressionHead(
        model_name=config["model_name"],
        unfreeze_layers=config["unfreeze_layers"],
        hidden_dim=config["dense"],
        dropout=config["dropout"],
        custom_weights_path=config["custom_weights_path"]
    ).to(DEVICE)
    
    # 2. Data loading
    collate = ESMBatchConverter(model.alphabet, max_len=config["max_len"])
    # Ensure data_dir path is correct
    train_loader = DataLoader(ProteinDataset(f"{config['data_dir']}/train.pkl"), 
                              batch_size=config["batch_size"], shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(ProteinDataset(f"{config['data_dir']}/valid.pkl"), 
                              batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers=0)
    test_loader  = DataLoader(ProteinDataset(f"{config['data_dir']}/test.pkl"), 
                              batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers=0)
    
    # 3. Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': config["weight_decay"]},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["lr"], eps=config["epsilon"])
    scaler = GradScaler()
    
    # 4. Training loop
    best_spearman = -1.0
    best_val_metrics = {}
    patience_counter = 0
    history_data = [] 
    
    epochs_range = range(1, config["epochs"] + 1)
    if verbose:
        print(f"[INFO] Start Training: {run_name} | Max Epochs: {config['epochs']}")
    
    for epoch in epochs_range:
        loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_metrics = evaluate(model, valid_loader)
        
        # Record logs
        history_data.append({
            "epoch": epoch,
            "train_loss": loss,
            "val_spearman": val_metrics["spearman"],
            "val_mse": val_metrics["mse"],
            "val_pearson": val_metrics["pearson"]
        })
        
        if verbose:
            print(f"  [Ep {epoch}] Loss: {loss:.4f} | Val Sp: {val_metrics['spearman']:.4f}")
        
        # Early stopping and saving logic
        if val_metrics["spearman"] > best_spearman:
            best_spearman = val_metrics["spearman"]
            best_val_metrics = val_metrics.copy()
            patience_counter = 0
            
            # === Core modification: Use lightweight saving ===
            # Must save file because test needs to load best weights later.
            # If in Grid mode, we decide whether to delete it outside run_experiment.
            save_trainable_model(model, save_path)
            
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                if verbose: print(f"Early stopping at epoch {epoch}")
                break
    
    # 5. Test (load best model)
    if os.path.exists(save_path):
        if verbose: print("  Running Test on Best Model...")
        # === Core modification: strict=False loading ===
        # Because file only contains partial parameters, must allow mismatch
        model.load_state_dict(torch.load(save_path), strict=False)
    else:
        print("Warning: No best model file found. Using last epoch.")
    
    test_metrics = evaluate(model, test_loader)
    
    # Save training curve data
    history_df = pd.DataFrame(history_data)
    history_file = os.path.join("results", f"{run_name}_curve.dat")
    history_df.to_csv(history_file, index=False, sep='\t')

    # Clean up memory
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    
    return test_metrics, best_val_metrics, history_data, save_path

# ===========================
# 5. Plotting Functions
# ===========================
def plot_dual_axis(history_data, run_name):
    epochs = [h["epoch"] for h in history_data]
    train_losses = [h["train_loss"] for h in history_data]
    val_losses = [h["val_mse"] for h in history_data] 
    spearmans = [h["val_spearman"] for h in history_data]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Left axis: Loss ---
    color_train = 'tab:red'
    color_val = 'tab:orange'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)', color=color_train)
    
    l1, = ax1.plot(epochs, train_losses, color=color_train, label='Train Loss')
    l2, = ax1.plot(epochs, val_losses, color=color_val, linestyle='--', label='Val Loss')
    
    ax1.tick_params(axis='y', labelcolor=color_train)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Right axis: Spearman ---
    ax2 = ax1.twinx() 
    color_sp = 'tab:blue'
    ax2.set_ylabel('Val Spearman', color=color_sp)
    l3, = ax2.plot(epochs, spearmans, color=color_sp, label='Val Spearman', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_sp)

    # --- Legend ---
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title(f'Training Curve: {run_name}')
    fig.tight_layout()
    plt.savefig(f"results/{run_name}_curve.png", dpi=300)
    plt.close()
    print(f"  [Plot] Saved to results/{run_name}_curve.png")

# ===========================
# 6. Main Program Entry
# ===========================
def pack_result_row(params, val_met, test_met):
    return {
        "epochs": params["epochs"],
        "learning_rate": params["lr"],
        "batch_size": params["batch_size"],
        "epsilon": params["epsilon"],
        "dropout": params["dropout"],
        "dense": params["dense"],
        "seed": params["seed"],
        "num_labels": params["num_labels"],
        "unfreeze_layers": params["unfreeze_layers"], 
        
        "val_spearman": val_met.get("spearman"),
        "val_pearson": val_met.get("pearson"),
        "val_mse": val_met.get("mse"),
        "val_rmse": val_met.get("rmse"),
        "val_mae": val_met.get("mae"),
        "val_r2": val_met.get("r2"),
        
        "test_spearman": test_met.get("spearman"),
        "test_pearson": test_met.get("pearson"),
        "test_mse": test_met.get("mse"),
        "test_rmse": test_met.get("rmse"),
        "test_mae": test_met.get("mae"),
        "test_r2": test_met.get("r2"),
    }

TARGET_COLUMNS = [
    "epochs", "learning_rate", "batch_size", "epsilon", "dropout", 
    "dense", "seed", "num_labels", "unfreeze_layers",
    "val_spearman", "val_pearson", "val_mse", "val_rmse", "val_mae", "val_r2",
    "test_spearman", "test_pearson", "test_mse", "test_rmse", "test_mae", "test_r2"
]

if __name__ == "__main__":
    # Configuration
    os.environ["TORCH_HOME"] = "/data/zxc" 
    os.makedirs("results", exist_ok=True)
    
    # Switch mode: "SINGLE" or "GRID"
    RUN_MODE = "GRID"  
    
    BASE_CONFIG = {
        "seed": 42,
        "model_name": "esm2_t33_650M_UR50D",
        "data_dir": "./GB1",
        "max_len": 1024,
        "custom_weights_path": None,
        # "custom_weights_path": "/data/zxc/human_proteome/GB1/weight/CL0041_0.9.pth",
        "weight_decay": 1e-2,
        "patience": 100,
        "epsilon": 1e-8,
        "num_labels": 1
    }
    
    # ==========================================
    # Mode A: Single Debug
    # ==========================================
    if RUN_MODE == "SINGLE":
        params = BASE_CONFIG.copy()
        params.update({
            "epochs": 10000,
            "batch_size": 256,
            "lr": 1e-4,
            "dropout": 0.1,
            "dense": 64,
            "unfreeze_layers": 1
        })
        
        run_name = "Single_Debug"
        print(f"\n RUNNING SINGLE EXPERIMENT: {run_name}")
        
        test_met, val_met, hist, saved_path = run_experiment(params, run_name=run_name)
        plot_dual_axis(hist, run_name)
        
        # Save CSV
        row = pack_result_row(params, val_met, test_met)
        out_csv = f"results/{run_name}_results.csv"
        df_curr = pd.DataFrame([row]).reindex(columns=TARGET_COLUMNS)
        df_curr.to_csv(out_csv, index=False)
        
        print("\n" + "="*40)
        print(f"Val Spearman:  {val_met.get('spearman', 0):.4f}")
        print(f"Test Spearman: {test_met.get('spearman', 0):.4f}")
        print(f"Model Size:    {os.path.getsize(saved_path)/1024/1024:.2f} MB (Optimized)")
        print("="*40)

    # ==========================================
    # Mode B: Grid Search
    # ==========================================
    elif RUN_MODE == "GRID":
        print("\n RUNNING GRID SEARCH...")
        
        grid_space = {
            "unfreeze_layers": [1],
            "dense": [16],
            "dropout": [0.1],
            "lr": [1e-3, 1e-4, 5e-4, 1e-5, 5e-5],
            "batch_size": [256]
        }
        
        keys, values = zip(*grid_space.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f" Total combinations: {len(combinations)}")
        
        all_results = []
        out_path = "results/grid_search_results.csv"
        
        # Record global best score for deciding file retention
        global_best_spearman = -1.0
        
        for i, combo in enumerate(combinations):
            params = BASE_CONFIG.copy()
            params.update(combo)
            params["epochs"] = 10000
            
            run_name = f"Grid_{i+1}"
            print(f"\n[{i+1}/{len(combinations)}] Params: {combo}")
            
            try:
                # Run experiment (internally auto-saves best .pth to disk)
                test_met, val_met, hist, saved_path = run_experiment(params, run_name=run_name, verbose=True)
                plot_dual_axis(hist, run_name) 

                # Record data
                row = pack_result_row(params, val_met, test_met)
                all_results.append(row)
                
                # Update summary CSV
                df_curr = pd.DataFrame(all_results).reindex(columns=TARGET_COLUMNS)
                df_curr = df_curr.sort_values(by="val_spearman", ascending=False)
                df_curr.to_csv(out_path, index=False)
                
                # === Core modification: File cleanup strategy ===
                current_spearman = val_met.get("spearman", -1)
                
                if current_spearman > global_best_spearman:
                    print(f"NEW GLOBAL BEST! ({current_spearman:.4f} > {global_best_spearman:.4f})")
                    print(f"Keeping model file: {saved_path}")
                    global_best_spearman = current_spearman
                else:
                    # If not global best, delete file to save space
                    if os.path.exists(saved_path):
                        os.remove(saved_path)
                        print(f"Result ({current_spearman:.4f}) not beating global best. Deleted {saved_path}")
                
            except Exception as e:
                print(f"Error in run {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue