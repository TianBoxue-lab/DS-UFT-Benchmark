import os,sys
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats
import itertools
from datetime import datetime

# ========== Set Random Seed ==========
def set_seed(s):
    tf.random.set_seed(s)
    np.random.seed(s)
    random.seed(s)

# ========== Read Data ==========
def read():
    df_train = pd.read_pickle("../../data/train_embedding.pkl")
    df_valid = pd.read_pickle("../../data/valid_embedding.pkl")
    df_test = pd.read_pickle("../../data/test_embedding.pkl")
    return df_train, df_valid, df_test


# ========== Custom Spearman Metric ==========
def spearmanr_metric(y_true, y_pred):
    y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_flat = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    corr, _ = tf.py_function(scipy.stats.spearmanr, [y_true_flat, y_pred_flat], [tf.float64, tf.float64])
    return corr

# ========== Build Model ==========
def emb_predictor(lr, dropout, dense, normalizer, epsilon, num_label=1):
    tf.keras.backend.clear_session()
    model = keras.Sequential([
        normalizer,
        layers.Dense(dense, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(num_label)
    ])

    if num_label == 1:
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon),
                      metrics=[spearmanr_metric])
    else:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon))
    
    return model


# ========== Plot and Save ==========
def plot_training_results(history, out_name):
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error [MSE]')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(history.history['val_spearmanr_metric'], label='val_spearman', color='r')
    ax2.set_ylabel('Spearman')
    ax2.set_ylim(0, 1)
    lns = ax1.lines + ax2.lines
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='center right')
    plt.title("Training Progress")
    plt.savefig(f"{out_name}.png")
    plt.close()


# ========== Validation or Test Evaluation ==========
def evaluate_and_save(model, x, y, name):
    y_pred = model.predict(x).flatten()
    spearman = scipy.stats.spearmanr(y, y_pred).correlation
    pearson = scipy.stats.pearsonr(y, y_pred)[0]
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    r2 = r2_score(y, y_pred)

    return spearman, pearson, mse, rmse, mae, r2


# ========== Training Function ==========
def train_predictor(epochs=10000, lr=1e-4, epsilon=1e-07, batch=8, dropout=0.2, dense=32, seed=99, num_labels=1):
    set_seed(seed)
    df_train, df_valid, df_test = read()

    x_train, y_train = np.array(df_train.iloc[:, :-2]), np.array(df_train.iloc[:, -1])
    x_valid, y_valid = np.array(df_valid.iloc[:, :-2]), np.array(df_valid.iloc[:, -1])
    x_test, y_test = np.array(df_test.iloc[:, :-2]), np.array(df_test.iloc[:, -1])

    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)

    model = emb_predictor(lr, dropout, dense, normalizer, epsilon, num_labels)
    model.summary()

    monitor_metric = 'val_loss'

    # Stop training when val_spearmanr_metric has not improved for 100 epochs
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=100,
        mode='min',
        restore_best_weights=True
    )

    # Only save model weights with highest Spearman coefficient on validation set
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="best_model_weights.h5",
        monitor=monitor_metric,
        mode='min',
        save_best_only=True,
        save_weights_only=True
    )
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        batch_size=batch,
        callbacks=[early_stopping, model_checkpoint],
        shuffle=True,
        verbose=1
    )

    best_loss = min(history.history[monitor_metric])
    best_epoch = np.argmin(history.history[monitor_metric])
    print(f"\n[OK] Training finished. Best '{monitor_metric}' = {best_loss:.4f} at epoch {best_epoch}")

    return model, history, (x_valid, y_valid), (x_test, y_test)    

# ========== Main Entry Point ==========
if __name__ == "__main__":

    # Single Test
    epochs = 10000
    lr = float(sys.argv[4]) # 1e-4
    batch = int(sys.argv[5]) # 128
    epsilon = 1e-07
    dropout = float(sys.argv[3]) # 0.2
    dense = int(sys.argv[2]) # 32
    seed = int(sys.argv[1]) # 42
    num_labels = 1

    model, history, val_data, test_data = train_predictor(epochs=epochs, lr=lr, epsilon=epsilon, batch=batch, dropout=dropout,
                                                          dense=dense, seed=seed, num_labels=num_labels)
    plot_training_results(history, "training_plot")

    val_spearman, val_pearson, val_mse, val_rmse, val_mae, val_r2 = evaluate_and_save(model, *val_data, name="valid")
    test_spearman, test_pearson, test_mse, test_rmse, test_mae, test_r2 = evaluate_and_save(model, *test_data, name="test")

    result = {
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch,
        "epsilon": epsilon,
        "dropout": dropout,
        "dense": dense,
        "seed": seed,
        "num_labels": num_labels,
        # val
        "val_spearman": val_spearman,
        "val_pearson": val_pearson,
        "val_mse": val_mse,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
        # test
        "test_spearman": test_spearman,
        "test_pearson": test_pearson,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2
    }

    pd.DataFrame(result, index=[0]).to_csv("results.csv", index=False)


    # # Hyperparameter Grid Search
    # dense_units_list = [32, 64]
    # dropout_rate_list = [0.2, 0.3]
    # learning_rate_list = [1e-3, 1e-4]
    # batch_size_list = [64, 128]
    
    # param_grid = list(itertools.product(
    #     dense_units_list,
    #     dropout_rate_list,
    #     learning_rate_list,
    #     batch_size_list
    # ))
    
    # total_combinations = len(param_grid)
    # print(f"[INFO] Starting Grid Search for {total_combinations} combinations...")
    
    # all_results = []
    
    # for i, params in enumerate(param_grid):
    #     dense, dropout, lr, batch = params
        
    #     print(f"\n[{i+1}/{total_combinations}] Running with: "
    #           f"dense={dense}, dropout={dropout}, lr={lr}, batch_size={batch}")
        
    #     # Use the same seed for reproducibility each time
    #     model, history, val_data, test_data = train_predictor(
    #         epochs=200,
    #         seed=42,
    #         dense=dense,
    #         dropout=dropout,
    #         lr=lr,
    #         batch=batch
    #     )

    #     # Evaluate validation and test sets
    #     val_spearman, val_pearson, val_mse, val_rmse, val_mae, val_r2 = evaluate_and_save(model, *val_data)
    #     test_spearman, test_pearson, test_mse, test_rmse, test_mae, test_r2 = evaluate_and_save(model, *test_data)
        
    #     print(f"  => Val Spearman: {val_spearman:.4f}, Test Spearman: {test_spearman:.4f}")

    #     # 4. Save parameters and results to dictionary
    #     current_result = {
    #         "dense": dense,
    #         "dropout": dropout,
    #         "learning_rate": lr,
    #         "batch_size": batch,
    #         # val
    #         "val_spearman": val_spearman,
    #         "val_pearson": val_pearson,
    #         "val_mse": val_mse,
    #         "val_rmse": val_rmse,
    #         "val_mae": val_mae,
    #         "val_r2": val_r2,
    #         # test
    #         "test_spearman": test_spearman,
    #         "test_pearson": test_pearson,
    #         "test_mse": test_mse,
    #         "test_rmse": test_rmse,
    #         "test_mae": test_mae,
    #         "test_r2": test_r2
    #     }
    #     all_results.append(current_result)

    # results_df = pd.DataFrame(all_results)
    # results_df = results_df.sort_values(by="val_spearman", ascending=False)    
    # results_df.to_csv('grid_search_results.csv', index=False)
    # print("\nTop 5 Best Parameter Combinations:")
    # print(results_df.head(5))
