import os
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
# class ResidueEmbeddingDataset(Dataset):
#     def __init__(self, dataframe):
#         self.df = dataframe
#         embedding_cols = [col for col in self.df.columns if col.startswith('emb_')]
#         if not embedding_cols:
#             raise ValueError("No embedding columns starting with 'emb_' found in DataFrame.")
#         self.embeddings = torch.tensor(self.df[embedding_cols].values, dtype=torch.float32)
#         self.labels = torch.tensor(self.df['label'].values, dtype=torch.long)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         return self.embeddings[idx], self.labels[idx]

class NpyDataset(Dataset):
    # cpu
    def __init__(self, embedding_path, label_path):
        self.embeddings = torch.from_numpy(np.load(embedding_path))
        self.labels = torch.from_numpy(np.load(label_path))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

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

# Model 2: 1D-CNN
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

# Model 3: BiLSTM
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

# Model 4: Transformer Encoder
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

# ========== Core Training and Evaluation Function ==========
def train_and_evaluate(params, model_type, train_df, valid_df, test_df, device, class_names, class_weights=None):
    set_seed(params['seed'])
    
    # Unpack parameters from params
    lr = params['lr']
    batch_size = params['batch_size']
    epochs = params['epochs']
    early_stopping_patience = params['early_stopping_patience']
    
    # Create DataLoader
    train_dataset = NpyDataset("./data/train_embeddings.npy", "./data/train_labels.npy")
    valid_dataset = NpyDataset("./data/valid_embeddings.npy", "./data/valid_labels.npy")
    test_dataset = NpyDataset("./data/test_embeddings.npy", "./data/test_labels.npy")

    num_workers = 8 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Initialize model, loss function and optimizer
    model = get_model(model_type, params, 1280, len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=lr)    

    # Variables for recording results
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    best_model_state = None
    train_loss_history, val_loss_history, val_acc_history = [], [], [] 

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = loss_function(logits, labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # Validation
        val_loss, val_accuracy, _, _, _ = evaluate(model, valid_loader, device, loss_function)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Use best model for final testing
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: Best model state not found, will use the model from the last epoch for testing.")

    test_loss, test_accuracy, test_labels, test_preds, test_scores = evaluate(model, test_loader, device, loss_function)

    report_dict = classification_report(
        test_labels, test_preds, target_names=class_names, digits=4, output_dict=True
    )
    n_classes = len(class_names)
    bina_labels = label_binarize(test_labels, classes=range(n_classes))
    macro_ap = average_precision_score(bina_labels, test_scores, average='macro')
    weighted_ap = average_precision_score(bina_labels, test_scores, average='weighted')

    return {
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "macro_f1": report_dict['macro avg']['f1-score'],
        "macro_precision": report_dict['macro avg']['precision'],
        "macro_recall": report_dict['macro avg']['recall'],
        "macro_ap": macro_ap,
        "weighted_f1": report_dict['weighted avg']['f1-score'],
        "weighted_precision": report_dict['weighted avg']['precision'],
        "weighted_recall": report_dict['weighted avg']['recall'],
        "weighted_ap": weighted_ap,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
        "test_labels": test_labels,
        "test_preds": test_preds,
        "test_scores": test_scores
    }

# ========== Grid Search Function ==========
def run_grid_search(params_config, model_type, train_df, valid_df, test_df, device, class_names, fixed_params, class_weights=None):
    grid_spec = {k: (v if isinstance(v, list) else [v]) for k, v in params_config.items()}
    keys, values = zip(*grid_spec.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Model type: {model_type.upper()} | Total {len(param_combinations)} hyperparameter combinations to run...")
    
    all_results = []
    start_time = time.time()

    for i, params in enumerate(param_combinations):
        # Merge fixed parameters with current combination parameters
        full_params = {**params, **fixed_params}

        print(f"\n[{i+1}/{len(param_combinations)}] Running: {params}")
        
        try:
            # Call core training function, only get required results
            results = train_and_evaluate(full_params, model_type, train_df, valid_df, test_df, device, class_names,class_weights)
            metrics_to_save = [
                'best_val_accuracy', 'test_accuracy', 'test_loss',
                'macro_f1', 'macro_precision', 'macro_recall', 'macro_ap',
                'weighted_f1', 'weighted_precision', 'weighted_recall', 'weighted_ap'
            ]

            current_result = params.copy()
            for m in metrics_to_save:
                current_result[m] = results.get(m, None)
            all_results.append(current_result)
            print(f"  => Result: Best validation accuracy = {results['best_val_accuracy']:.4f}, Test accuracy = {results['test_accuracy']:.4f}")

        except Exception as e:
            print(f"  => (!) Run failed: {params}\n     Error message: {e}")
            current_result = params.copy()
            current_result['best_val_accuracy'] = 0.0
            current_result['test_accuracy'] = 0.0
            current_result['error'] = str(e)
            all_results.append(current_result)

    end_time = time.time()
    print(f"\nGrid search completed, total time: {(end_time - start_time)/60:.2f} minutes.")
    
    if all_results:
        results_df = pd.DataFrame(all_results).sort_values(by="best_val_accuracy", ascending=False)
        output_filename = f'grid_search_{model_type}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"\nGrid search results saved to '{output_filename}'")
        print("\n--- Top 5 best performing hyperparameter sets ---")
        print(results_df.head(5))

# ========== Result Saving and Plotting Function ==========
def plot_and_save_loss_curve(train_loss, val_loss, val_accuracy, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    # Plot loss curve (left Y-axis)
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_loss, color=color, linestyle='--', label='Training Loss')
    ax1.plot(val_loss, color=color, linestyle='-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle=':', color=color)  # Add dashed grid only for left axis
    # Create second Y-axis, sharing X-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(val_accuracy, color=color, marker='o', linestyle='-', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legend
    # To display legends from both lines together, we need to get them from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Training & Validation Loss and Accuracy')
    fig.tight_layout()  # Adjust layout to prevent label overlap
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved to: {save_path}")

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

if __name__ == '__main__':
    MODEL_TO_RUN = 'MLP'  # 'MLP', 'CNN', 'BiLSTM', 'Transformer'

    # --- Single Run Parameters ---
    # To perform a single run, set is_grid_search to False
    SINGLE_RUN_PARAMS = {
        'lr': 1e-4,
        'batch_size': 512,
        'dropout': 0.1,
        'dense_units': 512,      # For MLP
        'num_filters': 128,      # For CNN
        'kernel_size': 7,        # For CNN
        'hidden_dim': 256,       # For BiLSTM
        'n_layers': 2,           # For BiLSTM
        'nhead': 8,              # For Transformer
        'num_encoder_layers': 2, # For Transformer
        'dim_feedforward': 2048, # For Transformer
    }
    
    # --- Grid Search Parameters ---
    # To perform grid search, set is_grid_search to True
    # Grid search MLP
    GRID_SEARCH_PARAMS_MLP = {
        'lr': [1e-3, 1e-4, 5e-4, 1e-5, 5e-5],
        'batch_size': [128, 256, 512, 1024], 
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'dense_units': [512],
    }

    # Grid search CNN
    GRID_SEARCH_PARAMS_CNN = {
        'lr': [1e-4, 5e-5],
        'batch_size': [256], 
        'dropout': [0.1],
        'num_filters': [64, 128],
        'kernel_size': [5, 7, 9],
    }

    # Grid search BiLSTM
    GRID_SEARCH_PARAMS_BiLSTM = {
        'lr': [1e-4, 5e-5],
        'batch_size': [256], 
        'dropout': [0.1],
        'hidden_dim': [128, 256],
        'n_layers': [1, 2],
    }

    # Grid search Transformer
    GRID_SEARCH_PARAMS_Transformer = {
        'lr': [1e-4, 5e-5],
        'batch_size': [256], 
        'dropout': [0.1],
        'nhead': [4, 8],
        'num_encoder_layers': [1, 2],
        'dim_feedforward': [1024, 2048],
    }

    # --- Fixed Settings ---
    SEED = 42
    EPOCHS = 1000
    EARLY_STOPPING_PATIENCE = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASS_NAMES = ['Coil (C)', 'Strand (E)', 'Helix (H)']

    # Decide run mode
    is_grid_search = True  # <--- Switch True/False here to select mode
    
    print(f"Using device: {DEVICE}")
    set_seed(SEED)
    print("Starting to load and process data...")

    train_df = pd.read_pickle("./data/train_embedding.pkl")
    valid_df = pd.read_pickle("./data/valid_embedding.pkl")
    test_df = pd.read_pickle("./data/test_embedding.pkl")
 
    def process_labels_vectorized(df):
        df = df.copy(); df.loc[df['mask'].astype(str) == '0', 'label'] = -100; return df
    train_df, valid_df, test_df = map(process_labels_vectorized, [train_df, valid_df, test_df])
    print("Data preparation complete.")

    print("\nStarting to calculate class weights...")
    train_labels_for_weight = train_df[train_df['label'] != -100]['label'].values
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels_for_weight),
        y=train_labels_for_weight
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Calculated class weights: {class_weights}")

    # Combine fixed parameters
    fixed_params = {
        'seed': SEED,
        'epochs': EPOCHS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE
    }

    # Single run mode
    if not is_grid_search:
        print(f"\n===== Starting【Single Run】Mode =====")
        print(f"模型类型: {MODEL_TO_RUN.upper()}")
        
        # Merge single run parameters and fixed parameters
        params = {**SINGLE_RUN_PARAMS, **fixed_params}
        print(f"运行参数: {params}")
        
        start_time = time.time()
        
        # Call core training function
        results = train_and_evaluate(
            params=params, model_type=MODEL_TO_RUN, train_df=train_df,
            valid_df=valid_df, test_df=test_df, device=DEVICE, class_names=CLASS_NAMES,
            class_weights=class_weights
        )
        
        end_time = time.time()
        
        # Create dedicated result folder for this run
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = f'results_single_run/{MODEL_TO_RUN}_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all results        
        plot_and_save_loss_curve(
            results['train_loss_history'], 
            results['val_loss_history'], 
            results['val_acc_history'],
            os.path.join(output_dir, 'training_curves.png')
        )
        save_full_report(results, params, CLASS_NAMES, output_dir)
        
        print("\n--- Single Run Result Summary ---")
        print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        print(f"Runtime: {(end_time - start_time):.2f} seconds")
        print(f"All detailed results saved to: {output_dir}")

    # Grid search mode
    else:
        print(f"\n===== Starting【Grid Search】Mode =====")
        # Here you can choose different grid search configurations based on MODEL_TO_RUN value
        if MODEL_TO_RUN == 'MLP':
            params_config = GRID_SEARCH_PARAMS_MLP
        elif MODEL_TO_RUN == 'CNN':
            params_config = GRID_SEARCH_PARAMS_CNN
        elif MODEL_TO_RUN == 'BiLSTM':
            params_config = GRID_SEARCH_PARAMS_BiLSTM
        elif MODEL_TO_RUN == 'Transformer':
            params_config = GRID_SEARCH_PARAMS_Transformer
        else:
            raise ValueError(f"No grid search parameter configuration defined for model {MODEL_TO_RUN}.")
            
        run_grid_search(
            params_config, MODEL_TO_RUN, train_df, valid_df, test_df, DEVICE, CLASS_NAMES, fixed_params, class_weights
        )
