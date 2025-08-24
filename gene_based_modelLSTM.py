"""
Imports
"""
#1. Import necessary libraries and functions
from shared_utils import (
    load_data,
    add_timepoint_data_to_adata,
    preprocess_data,
    add_qc_metrics,
    plot_before_filtering,
    create_gene_time_matrix,
    split_by_genes,
    create_gene_windows,
    build_lstm_model,
    build_feedforward_model,
    get_training_callbacks,
    evaluate_performance,
    plot_training_history,
    plot_prediction_comparison,
    plot_diagnostic_plots,
    plot_performance_comparison_bar_chart
)

def main():

    # 2. Load and preprocess the data
    adata = load_data()
    adata = add_timepoint_data_to_adata(adata)
    add_qc_metrics(adata)
    plot_before_filtering(adata)
    adata_hvg = preprocess_data(adata)
    gene_time_matrix, _ = create_gene_time_matrix(adata_hvg)

    # 3. Split data by genes and create windows
    train_data, val_data, test_data = split_by_genes(gene_time_matrix)
    
    # The window size is 5 to predict the 6th time point
    X_train, y_train = create_gene_windows(train_data, window_size=5, prediction_steps=1)
    X_val, y_val = create_gene_windows(val_data, window_size=5, prediction_steps=1)
    X_test, y_test = create_gene_windows(test_data, window_size=5, prediction_steps=1)


    input_shape = (X_train.shape[1], X_train.shape[2]) 
    output_size = y_train.shape[1]

    # 4. Build the LSTM and Feedforward models for comparison
    lstm_model = build_lstm_model(input_shape, output_size)
    ff_model = build_feedforward_model(input_shape, output_size)
    
    print("\n--- Training LSTM Model ---")
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=250,
        verbose=1,
        callbacks=get_training_callbacks()
    )
    print(f"LSTM training completed after {len(lstm_history.history['loss'])} epochs.")

    print("\n--- Training Feedforward Model ---")
    ff_history = ff_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=250,
        verbose=1,
        callbacks=get_training_callbacks()
    )
    print(f"Feedforward training completed after {len(ff_history.history['loss'])} epochs.")
    
    # 5. Evaluate models on the test set
    print("\n--- Final LSTM Model Evaluation ---")
    lstm_rmse, lstm_mae, lstm_r2, y_pred_lstm = evaluate_performance(y_test, lstm_model.predict(X_test))

    print("\n--- Final Feedforward Model Evaluation ---")
    ff_rmse, ff_mae, ff_r2, y_pred_ff = evaluate_performance(y_test, ff_model.predict(X_test))

    # 6. Plot results for visualization
    print("\n--- Plotting Training Histories ---")
    plot_training_history(lstm_history, model_name="LSTM")
    plot_training_history(ff_history, model_name="Feedforward")

    print("\n--- Plotting Prediction Comparisons ---")
    lstm_metrics = {'RMSE': lstm_rmse, 'MAE': lstm_mae, 'R²': lstm_r2}
    plot_prediction_comparison(y_test, y_pred_lstm, model_type='gene', metrics_dict=lstm_metrics)

    ff_metrics = {'RMSE': ff_rmse, 'MAE': ff_mae, 'R²': ff_r2}
    plot_prediction_comparison(y_test, y_pred_ff, model_type='gene', metrics_dict=ff_metrics)
    
    print("\n--- Plotting Diagnostic Plots ---")
    plot_diagnostic_plots(y_test, y_pred_lstm, model_name="LSTM")
    plot_diagnostic_plots(y_test, y_pred_ff, model_name="Feedforward")

    print("\n--- Plotting Final Performance Comparison ---")
    lstm_metrics = {'RMSE': lstm_rmse, 'MAE': lstm_mae, 'R²': lstm_r2}
    ff_metrics = {'RMSE': ff_rmse, 'MAE': ff_mae, 'R²': ff_r2}
    plot_performance_comparison_bar_chart(lstm_metrics, ff_metrics)


if __name__ == "__main__":
    main()