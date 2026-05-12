import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import itertools
import time
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.preprocessing import label_binarize

# ========== Set Random Seed ==========
def set_seed(seed=99):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The following two lines are typically used to ensure determinism, but may affect performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ========== Dataset Definition ==========
class NpyDataset(Dataset):
    # cpu
    def __init__(self, embedding_path, label_path):
        self.embeddings = torch.from_numpy(np.load(embedding_path))
        self.labels = torch.from_numpy(np.load(label_path))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ========== Model Definition (MLP, CNN, BiLSTM, Transformer) ==========
class MLP(nn.Module):
    def __init__(self, input_dim=1280, num_labels=3, dropout_prob=0.1, dense_units=512):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(input_dim, num_labels)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, embeddings):
        # embeddings: (batch_size, input_dim)
        x = self.dropout(embeddings)
        logits = self.classifier(x)
        return logits


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_labels=3, num_filters=128, kernel_size=7, dropout_prob=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(num_filters, num_labels)

    def forward(self, embeddings):
        x = embeddings.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_labels=3, hidden_dim=256, n_layers=2, dropout_prob=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout_prob if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, embeddings):
        x = embeddings.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=1280, num_labels=3, nhead=8, num_encoder_layers=2, dim_feedforward=2048, dropout_prob=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, 
                                                   dropout=dropout_prob, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.out_proj = nn.Linear(input_dim, num_labels)

    def forward(self, embeddings):
        x = embeddings.unsqueeze(1)
        encoded = self.transformer_encoder(x)
        output = encoded[:, 0, :]
        logits = self.out_proj(output)
        return logits


# ========== Model Factory ==========
def get_model(model_type, model_params, input_dim, num_labels):
    if model_type == 'MLP':
        return MLP(
            input_dim=input_dim,
            num_labels=num_labels,
            dropout_prob=model_params.get('dropout', 0.1),
            dense_units=model_params.get('dense_units', 512)
        )
    elif model_type == 'CNN':
        return CNN1DClassifier(
            input_dim=input_dim,
            num_labels=num_labels,
            num_filters=model_params.get('num_filters', 128),
            kernel_size=model_params.get('kernel_size', 7),
            dropout_prob=model_params.get('dropout', 0.3)
        )
    elif model_type == 'BiLSTM':
        return BiLSTMClassifier(
            input_dim=input_dim,
            num_labels=num_labels,
            hidden_dim=model_params.get('hidden_dim', 256),
            n_layers=model_params.get('n_layers', 2),
            dropout_prob=model_params.get('dropout', 0.3)
        )
    elif model_type == 'Transformer':
        return TransformerClassifier(
            input_dim=input_dim,
            num_labels=num_labels,
            nhead=model_params.get('nhead', 8),
            num_encoder_layers=model_params.get('num_encoder_layers', 2),
            dim_feedforward=model_params.get('dim_feedforward', 2048),
            dropout_prob=model_params.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ========== Strict SOV Calculation Function (Adapted to Academic Standards) ==========
def calculate_sov(true_labels, pred_labels):
    """
    Strict SOV'99 implementation. Uses unique matching logic to prevent a true segment from being scored multiple times by different predicted segments.
    """
    def get_segments(labels):
        segments = []
        if len(labels) == 0: return segments
        start = 0
        curr_label = labels[0]
        for i in range(1, len(labels)):
            if labels[i] != curr_label:
                segments.append({'label': curr_label, 'start': start, 'end': i - 1})
                start = i
                curr_label = labels[i]
        segments.append({'label': curr_label, 'start': start, 'end': len(labels) - 1})
        return segments

    true_segs = get_segments(true_labels)
    pred_segs = get_segments(pred_labels)
    
    # Dynamically obtain valid evaluation labels (avoid -100)
    labels_to_eval = np.unique(true_labels[true_labels != -100])
    
    total_sov_sum = 0
    total_residues_n = 0

    for label in labels_to_eval:
        S_obs = [s for s in true_segs if s['label'] == label]
        S_prd = [s for s in pred_segs if s['label'] == label]
        
        if not S_obs: continue
        
        N_i = sum([(s['end'] - s['start'] + 1) for s in S_obs])
        sum_val_i = 0
        
        for s_obs in S_obs:
            l_obs = s_obs['end'] - s_obs['start'] + 1
            # Find all overlapping predicted segments with consistent labels
            overlaps = []
            for s_prd in S_prd:
                ov_start = max(s_obs['start'], s_prd['start'])
                ov_end = min(s_obs['end'], s_prd['end'])
                if ov_start <= ov_end:
                    overlaps.append((s_prd, ov_end - ov_start + 1))
            
            if not overlaps: continue

            # --- Key matching logic: Select the predicted segment with the longest overlap as the unique match ---
            best_s_prd, best_ov_len = max(overlaps, key=lambda x: x[1])
            
            l_prd = best_s_prd['end'] - best_s_prd['start'] + 1
            minov = best_ov_len
            maxov = max(best_s_prd['end'], s_obs['end']) - min(best_s_prd['start'], s_obs['start']) + 1
            
            # Tolerance factor delta
            delta = min(maxov - minov, minov, l_obs // 2, l_prd // 2)
            sum_val_i += ((minov + delta) / maxov) * l_obs
        
        total_sov_sum += sum_val_i
        total_residues_n += N_i

    return (100.0 * total_sov_sum / total_residues_n) if total_residues_n > 0 else 0.0


# ========== Evaluation Function ==========
def evaluate(model, data_loader, device, loss_function):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            logits = model(embeddings)

            scores = torch.softmax(logits, dim=1)
            loss = loss_function(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            valid_mask = labels != -100
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_labels.extend(labels[valid_mask].cpu().numpy())
            all_scores.extend(scores[valid_mask].cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_labels, all_preds, all_scores


# ========== Result Report Saving ==========
def save_full_report(results, params, class_names, output_dir):
    """
    Save a comprehensive evaluation report, including:
    1. A CSV file containing all core metrics.
    2. A TXT file containing detailed parameters and text report.
    3. A PNG image of the confusion matrix.
    """
    test_labels = results['test_labels']
    test_preds = results['test_preds']
    test_scores = results['test_scores']
    n_classes = len(class_names)

    # --- 1. Save core metrics to CSV file ---
    csv_data = params.copy()
    csv_data.update({
        'best_val_accuracy': results['best_val_accuracy'],
        'test_accuracy': results['test_accuracy'],
        "test_sov": results['test_sov'],
        'macro_precision': results['macro_precision'],
        'macro_recall': results['macro_recall'],
        'macro_f1': results['macro_f1'],
        'macro_ap': results['macro_ap'],
        'weighted_precision': results['weighted_precision'],
        'weighted_recall': results['weighted_recall'],
        'weighted_f1': results['weighted_f1'],
        'weighted_ap': results['weighted_ap'],
    })
    
    csv_path = os.path.join(output_dir, 'summary_metrics.csv')
    pd.DataFrame([csv_data]).to_csv(csv_path, index=False)
    print(f"Core evaluation metrics saved to: {csv_path}")

    # --- 2. Save comprehensive report with text and images to TXT file ---
    # Generate text version of classification report string
    report_str = classification_report(
        test_labels, 
        test_preds, 
        target_names=class_names,
        digits=4
    )

    # Generate text version of confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    cm_str = np.array2string(cm, separator=', ')

    # Prepare complete content to be written to TXT file
    full_report_content = f"""
=================================================
       Full Evaluation Report
=================================================

# 1. Run Parameters
-------------------
{pd.Series(params).to_string()}

# 2. Key Metrics Summary
------------------------
Best Val Accuracy:      {results['best_val_accuracy']:.4f}
Test Accuracy:          {results['test_accuracy']:.4f}
Test Loss:              {results['test_loss']:.4f}
Test SOV Score:         {results['test_sov']:.4f}
Macro F1:               {results['macro_f1']:.4f}
Macro Precision:        {results['macro_precision']:.4f}
Macro Recall:           {results['macro_recall']:.4f}
Macro AP:               {results['macro_ap']:.4f}
Weighted F1:            {results['weighted_f1']:.4f}
Weighted Precision:     {results['weighted_precision']:.4f}
Weighted Recall:        {results['weighted_recall']:.4f}
Weighted AP:            {results['weighted_ap']:.4f}

# 3. Full Classification Report
-------------------------------
{report_str}

# 4. Confusion Matrix
---------------------
{cm_str}
(Corresponding class order: {class_names})

# 5. Visualizations
-------------------
- Loss/Accuracy curves saved to 'training_curves.png'
- Confusion matrix plot saved to 'confusion_matrix.png'
"""
    
    report_txt_path = os.path.join(output_dir, 'full_evaluation_report.txt')
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(full_report_content)
    print(f"Detailed text report saved to: {report_txt_path}")

    # --- 3. Save confusion matrix plot (logic unchanged) ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")


# ========== Main Program Entry ==========
if __name__ == '__main__':
    MODEL_TO_RUN = 'MLP'  # 'MLP', 'CNN', 'BiLSTM', 'Transformer'

    # --- Single Run Parameters ---
    # To perform a single run, set is_grid_search to False
    SINGLE_RUN_PARAMS = {
        'lr': float(sys.argv[2]),
        'batch_size': int(sys.argv[3]),
        'dropout': float(sys.argv[4]),
        'dense_units': 512,      # For MLP
        'num_filters': 128,      # For CNN
        'kernel_size': 7,        # For CNN
        'hidden_dim': 256,       # For BiLSTM
        'n_layers': 2,           # For BiLSTM
        'nhead': 8,              # For Transformer
        'num_encoder_layers': 2, # For Transformer
        'dim_feedforward': 2048, # For Transformer
    }
    
    # --- Fixed Settings ---
    SEED = int(sys.argv[1])
    MODEL_PATH = sys.argv[5]  # New: Pass the full path of the .pt file
    EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASS_NAMES = ['Coil (C)', 'Strand (E)', 'Helix (H)']

    # Decide run mode
    is_grid_search = False  # <--- Switch True/False here to select mode
    
    print(f"Using device: {DEVICE}")
    set_seed(SEED)
    print("Starting to load and process data...")

    # Load evaluation data
    # Need validation set to re-obtain 'best_val_accuracy'
    valid_dataset = NpyDataset("../data/valid_embeddings.npy", "../data/valid_labels.npy")
    test_dataset = NpyDataset("../data/test_embeddings.npy", "../data/test_labels.npy")
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Prepare class weights (to maintain consistency in Loss calculation)
    train_df = pd.read_pickle("../data/train_embedding.pkl")
    train_df.loc[train_df['mask'].astype(str) == '0', 'label'] = -100

    print("\nStarting to calculate class weights...")
    train_labels_for_weight = train_df[train_df['label'] != -100]['label'].values
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels_for_weight),
        y=train_labels_for_weight
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Calculated class weights: {class_weights}")

    loss_function = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # Combine fixed parameters
    fixed_params = {
        'seed': SEED,
        'epochs': EPOCHS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE
    }

    # Create dedicated result folder for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f'results/{MODEL_TO_RUN}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Merge single run parameters and fixed parameters
    params = {**SINGLE_RUN_PARAMS, **fixed_params}
    print(f"Run parameters: {params}")
    
    # Initialize model and load weights
    model = get_model(MODEL_TO_RUN, params, 1280, len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Perform re-evaluation
    print(f"Starting to re-evaluate weights: {MODEL_PATH}")
    best_val_loss, best_val_accuracy, _, _, _ = evaluate(model, valid_loader, DEVICE, loss_function)
    test_loss, test_accuracy, test_labels, test_preds, test_scores = evaluate(model, test_loader, DEVICE, loss_function)

    # Calculate SOV
    report_dict = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    sov_score = calculate_sov(np.array(test_labels), np.array(test_preds))
    n_classes = len(CLASS_NAMES)
    bina_labels = label_binarize(test_labels, classes=range(n_classes))
    macro_ap = average_precision_score(bina_labels, test_scores, average='macro')
    weighted_ap = average_precision_score(bina_labels, test_scores, average='weighted')

    results = {
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_sov": sov_score,
        "test_loss": test_loss,
        "macro_f1": report_dict['macro avg']['f1-score'],
        "macro_precision": report_dict['macro avg']['precision'],
        "macro_recall": report_dict['macro avg']['recall'],
        "macro_ap": macro_ap,
        "weighted_f1": report_dict['weighted avg']['f1-score'],
        "weighted_precision": report_dict['weighted avg']['precision'],
        "weighted_recall": report_dict['weighted avg']['recall'],
        "weighted_ap": weighted_ap,
        "test_labels": test_labels,
        "test_preds": test_preds,
        "test_scores": test_scores
    }

    # Save report
    output_dir = os.path.dirname(MODEL_PATH)
    save_full_report(results, params, CLASS_NAMES, output_dir)
    print(f"Re-evaluation completed. Results saved to: {output_dir}")
