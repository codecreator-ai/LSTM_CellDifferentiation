"""
Shared utility functions for the cell differentiation prediction model.

Contains the following components for both models:
1.  Data Loading and Preprocessing
2.  Data Splitting and Windowing
3.  Model Building (LSTM and Feedforward)
4.  Model Training, Evaluation, and Plotting
"""

# 1. IMPORTS
import math
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 2. DATA LOADING AND PREPROCESSING FUNCTIONS

def load_data(csv_path="GSE75748_sc_time_course_ec.csv"):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0)
    adata = sc.AnnData(df.T)
    return adata

def extract_timepoint(cell_name):
    if "hb4s" in cell_name:
        hour = cell_name.split(".")[1].split("hb4s")[0]
        samplNum = cell_name.split(".")[1].split("hb4s_")[1]
        return (hour, samplNum)
    else:
        hour = cell_name.split(".")[1].split("h_")[0]
        samplNum = cell_name.split(".")[1].split("h_")[1]
        return (hour, samplNum)
    return None

def add_timepoint_data_to_adata(adata):
    print("Extracting timepoints from cell names to add to adata")
    time_info = [extract_timepoint(name) for name in adata.obs_names]
    adata.obs["timepoint"] = [item[0] for item in time_info]
    adata.obs["sample_number"] = [item[1] for item in time_info]
    adata.obs["timepoint"] = [int(numval) for numval in adata.obs["timepoint"]]
    print("Timepoint data added to adata.")
    return adata

"""
Quality control metrics, so that we can filter cells and genes later without any problems.
"""

def add_qc_metrics(adata):
    adata.var_names_make_unique()
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    print("QC metrics added to adata.obs:")
    qc_columns = [col for col in adata.obs.columns if any(x in col for x in ['n_genes', 'total_counts', 'pct_counts'])]
    print(qc_columns)
    adata.obs
    return adata

"""
Violin plotting same from notebook
"""

def plot_before_filtering(adata):
    print("Plotting before filtering...")

    # First plot
    sc.pl.violin(
        adata,
        'n_genes_by_counts',
        jitter=0.4
    )

    # Second plot
    #sc.pl.violin(
    #    adata,
    #    'total_counts',
    #    jitter=0.4
    #)

"""
Doing normal filtering, normalization, log transformation, and HVG (Highly variable genes) selection
"""

def preprocess_data(adata, min_genes=500, max_genes=12000, min_cells=3, target_sum=1e4, n_top_genes=2000):
    print("Starting data preprocessing")
    print("Filtering cells and genes\n")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Data shape after filtering: {adata.shape}")
    print("\nNormalizing and log transforming data")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    print("\n\nHVG filtering:")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    print(f"Highly variable genes found: {adata.var['highly_variable'].sum()}") #should be n_top_genes
    adata = adata[:, adata.var.highly_variable]
    print(f"After HVG filtering:\n")
    time_points = sorted(adata.obs['timepoint'].unique())
    print(time_points)
    return adata

"""
Create matrix of genes and times so that it is easy to split by genes or by timepoints. It just simplifies everything, so I decided to keep this from my notebook
"""

def create_gene_time_matrix(adata):
    print("Creating gene-time matrix")
    time_points = sorted(adata.obs['timepoint'].unique())
    n_genes = adata.n_vars
    n_timepoints = len(time_points)
    gene_time_matrix = np.zeros((n_genes, n_timepoints))

    for i, timepoint in enumerate(time_points):
        cells_at_time = adata.obs["timepoint"] == timepoint
        expression_at_time = adata.X[cells_at_time, :]
        avg_expression = expression_at_time.mean(axis=0)
        if hasattr(avg_expression, 'A1'):
            avg_expression = avg_expression.A1
        gene_time_matrix[:, i] = avg_expression
        print(f"  Expression data shape: {expression_at_time.shape}")
        print(f"  First three values in row: {avg_expression[:3]}...")  # show first 3 genes
        print()

    print(f"Gene-time matrix fully created with shape: {gene_time_matrix.shape}")
    return gene_time_matrix, time_points

# 3. DATA SPLITTING AND WINDOWING FUNCTIONS

"""
Copied and pasted much of the below from the notebook
"""

def split_by_genes(gene_time_matrix, split_ratios=(0.7, 0.15, 0.15)):
    n_genes = gene_time_matrix.shape[0]
    train_ratio, val_ratio, _ = split_ratios
    train_end = int(train_ratio * n_genes)
    val_end = int((train_ratio + val_ratio) * n_genes)

    train_data = gene_time_matrix[:train_end, :]
    val_data = gene_time_matrix[train_end:val_end, :]
    test_data = gene_time_matrix[val_end:, :]

    print(f"Gene splitting: {n_genes} genes total")
    print(f"Train: {train_data.shape[0]} genes, Val: {val_data.shape[0]} genes, Test: {test_data.shape[0]} genes")
    return train_data, val_data, test_data

def split_by_cells(gene_time_matrix):
   
    train_data = gene_time_matrix[:, :4]  # Timepoints 0, 12, 24, 36
    val_data = gene_time_matrix[:, 2:5]   # Timepoints 24, 36, 72
    test_data = gene_time_matrix[:, 3:]   # Timepoints 36, 72, 96

    print(f"Temporal holdout splitting: {gene_time_matrix.shape[1]} timepoints total")
    print(f"Train: timepoints 0-3 ({train_data.shape[1]} points)")
    print(f"Val: timepoints 2-4 ({val_data.shape[1]} points)")
    print(f"Test: timepoints 3-5 ({test_data.shape[1]} points)")
    return train_data, val_data, test_data

def create_gene_windows(gene_data, window_size=5, prediction_steps=1):
    X, y = [], []
    n_genes, n_timepoints = gene_data.shape

    for gene_i in range(n_genes):
        gene_timeseries = gene_data[gene_i, :]
        if n_timepoints >= window_size + prediction_steps:
            window_x = gene_timeseries[:window_size]
            window_y = gene_timeseries[window_size : window_size + prediction_steps]
            X.append(window_x)
            y.append(window_y)

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, prediction_steps)

    print(f"Gene windows created: {X.shape[0]} samples. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def create_cell_windows(cell_data, window_size=2, prediction_steps=1):
    # Transpose to have timepoints as the first dimension
    cell_data = cell_data.T  # Shape: (n_timepoints, n_genes)
    n_timepoints, _ = cell_data.shape
    X, y = [], []

    for i in range(n_timepoints - window_size - prediction_steps + 1):
        window_x = cell_data[i : i + window_size, :]
        window_y = cell_data[i + window_size : i + window_size + prediction_steps, :]
        X.append(window_x)
        y.append(window_y.squeeze())

    if len(X) == 0:
        raise ValueError("Cannot create windows. Not enough timepoints for the given window/prediction size.")

    X, y = np.array(X), np.array(y)
    print(f"Cell windows created: {X.shape[0]} samples. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# 4. MODEL BUILDING FUNCTIONS

def build_feedforward_model(input_shape, output_size, dropout_rate=0.5):
    print(f"Building feedforward model: Input shape={input_shape}, Output size={output_size}")
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512 if output_size > 1 else 8, activation='relu'),
        Dropout(dropout_rate),
        Dense(output_size, activation='linear')
    ])
    model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
    return model

def build_lstm_model(input_shape, output_size, dropout_rate=0.2):
    print(f"Building LSTM model: Input shape={input_shape}, Output size={output_size}")

    if output_size == 1:  # Gene model
        lstm_units = [128, 64]
        dense_units = 32
    else:  #cell model
        lstm_units = [512, 256]
        dense_units = 1024

    model = Sequential([
        LSTM(lstm_units[0], return_sequences=True, dropout=dropout_rate,
             recurrent_dropout=dropout_rate, input_shape=input_shape),
        LSTM(lstm_units[1], return_sequences=False, dropout=dropout_rate,
             recurrent_dropout=dropout_rate),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(output_size, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# 5. MODEL TRAINING AND EVALUATION FUNCTIONS

def get_training_callbacks(patience=15, factor=0.5, min_lr=0.0001):
    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience // 2, min_lr=min_lr)
    ]

def evaluate_performance(y_test, y_pred):
    
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    rmse = math.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
    mae = mean_absolute_error(y_test_arr, y_pred_arr)
    ss_res = np.sum((y_test_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_test_arr - np.mean(y_test_arr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    return rmse, mae, r2, y_pred_arr

def plot_training_history(history, model_name=""):
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_prediction_comparison(y_test, y_pred, model_type, metrics_dict):
    rmse, mae, r2 = metrics_dict['RMSE'], metrics_dict['MAE'], metrics_dict['R²']
    plt.figure(figsize=(12, 6))

    if model_type == "gene":
        n_samples = min(100, len(y_test))
        indices = range(n_samples)
        plt.plot(indices, y_test[:n_samples].flatten(), label='Actual', color='blue', alpha=0.8)
        plt.plot(indices, y_pred[:n_samples].flatten(), label='Predicted', color='red', alpha=0.8)
        plt.title('Gene Expression Prediction Comparison', fontsize=14)
        plt.xlabel('Test Sample Index (Gene Windows)', fontsize=12)
        plt.ylabel('Gene Expression Value', fontsize=12)
    else: # cell model
        if y_test.ndim > 1 and y_test.shape[0] > 1: # Multiple prediction windows
            y_test_mean = np.mean(y_test, axis=1)
            y_pred_mean = np.mean(y_pred, axis=1)
            indices = range(len(y_test_mean))
            plt.plot(indices, y_test_mean, label='Actual (avg)', color='blue', alpha=0.8)
            plt.plot(indices, y_pred_mean, label='Predicted (avg)', color='red', alpha=0.8)
            plt.title('Cell State Prediction (Average Expression)', fontsize=14)
            plt.xlabel('Test Window Index', fontsize=12)
            plt.ylabel('Average Gene Expression', fontsize=12)
        else: # Single prediction window
            indices = range(min(100, y_test.shape[-1]))
            plt.plot(indices, y_test.flatten()[:len(indices)], label='Actual', marker='o', markersize=3, alpha=0.8)
            plt.plot(indices, y_pred.flatten()[:len(indices)], label='Predicted', marker='s', markersize=3, alpha=0.8)
            plt.title('Single Cell State Prediction (First 100 Genes)', fontsize=14)
            plt.xlabel('Gene Index', fontsize=12)
            plt.ylabel('Gene Expression Value', fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.tight_layout()
    plt.show()

"""
Extra plots below to give more things to talk on for the paper
"""

def plot_diagnostic_plots(y_test, y_pred, model_name="Model"):
    plt.figure(figsize=(12, 5))
    
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    # 1.) Scatter plot of predicted vs actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5, label='Predictions')
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: Predicted vs. Actual")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2.) Histogram of prediction errors
    plt.subplot(1, 2, 2)
    errors = y_test_flat - y_pred_flat
    plt.hist(errors, bins=50)
    plt.xlabel("Prediction Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{model_name}: Error Distribution")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_performance_comparison_bar_chart(lstm_metrics, ff_metrics):
    metrics = ['RMSE', 'MAE', 'R²']
    lstm_scores = [lstm_metrics[m] for m in metrics]
    ff_scores = [ff_metrics[m] for m in metrics]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, lstm_scores, width, label='LSTM', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, ff_scores, width, label='Feedforward', color='red', alpha=0.7)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels on top of the bars
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')

    fig.tight_layout()
    plt.show()