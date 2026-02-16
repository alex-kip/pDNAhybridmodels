# demo_training.py
"""
Demo script for training and testing E. coli fermentation models
All parameters are configurable at the top of the script
"""

import os
import sys
import pandas as pd
import torch


# Add the directory containing kernel.py to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from kernel
from kernel import (
    train_cv_ensemble_with_parallel,
    predict_with_ensemble, plot_prediction_with_variation,
    calculate_and_plot_mae, save_ensemble
)

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# ============================================================================

# ----------------------------
# Data Configuration
# ----------------------------
EXPERIMENTS_TO_TRAIN = ['1', '2', '3', '5']
EXPERIMENTS_TO_TEST = ['4']

# ----------------------------
# Model Architecture
# ----------------------------
MODEL_TYPE = 'fnn'  # 'fnn' or 'transformer'
HISTORY_STEPS = 3  # Number of historical steps to consider
MODEL_HIDDEN_WIDTH = 32  # Width of hidden layers
MODEL_HIDDEN_DEPTH = 4  # Number of hidden layers

# ----------------------------
# Training Configuration
# ----------------------------
MAX_EPOCHS = 100  # Maximum number of epochs
LEARNING_RATE = 1e-2  # Initial learning rate
OPTIMIZER_TYPE = 'adam'  # 'adam', 'sgd', or 'adamw'
USE_REDUCE_ON_PLATEAU = True  # Use ReduceLROnPlateau instead of CyclicLR
PREFER_MIN_SUM_LOSS = False  # If True, select model based on train+val loss sum


# ----------------------------
# Data Augmentation
# ----------------------------
USE_AUGMENTATION = False  # Enable data augmentation
NUM_AUGMENTED_POINTS = 100  # Number of points for augmentation
LAMBDA_SMOOTH = 1e-3  # Smoothness regularization weight

# ----------------------------
# ODE Solver Configuration
# ----------------------------
ODE_METHOD = 'euler'  # 'euler', 'rk4', 'dopri5', etc.
ODE_RTOL = 1e-4  # Relative tolerance
ODE_ATOL = 1e-5  # Absolute tolerance

# ----------------------------
# Optimization
# ----------------------------
MAX_GRAD_NORM = 1.0  # Gradient clipping norm
N_JOBS = 4  # Number of parallel jobs (-1 = all cores)

# ----------------------------
# Cross-Validation
# ----------------------------
N_SPLITS = None  # Number of CV splits (None = auto)



# ----------------------------
# Plotting Configuration
# ----------------------------

OUTPUT_DIR = "demo_results"  # Directory for saving results

# ----------------------------
# Model Saving/Loading
# ----------------------------
SAVE_MODEL = True  # Save trained model



# ============================================================================
# MAIN SCRIPT
# ============================================================================

def load_experimental_data():
    """
    Load experimental data from Excel file.
    Modify this function based on your data structure.
    """
    from kernel import EXCEL_FILE

    print(f"Loading data from: {EXCEL_FILE}")



    # Read all sheets
    xls = pd.ExcelFile(EXCEL_FILE)
    All_Data_Exp = {}

    for sheet_name in xls.sheet_names:
        if sheet_name.isdigit() or sheet_name in EXPERIMENTS_TO_TRAIN + EXPERIMENTS_TO_TEST:
            df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
            # Standardize column names
            df.columns = [col.strip() for col in df.columns]
            if 'Time' in df.columns:
                All_Data_Exp[sheet_name] = df

    print(f"Loaded {len(All_Data_Exp)} experiments")
    return All_Data_Exp

def main():
    """Main training and testing script."""

    print("=" * 70)
    print("E. COLI FERMENTATION MODEL TRAINING DEMO")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n📊 Loading experimental data...")
    All_Data_Exp = load_experimental_data()

    # Verify experiments exist
    available_exps = list(All_Data_Exp.keys())
    train_ids = [exp for exp in EXPERIMENTS_TO_TRAIN if exp in available_exps]
    test_ids = [exp for exp in EXPERIMENTS_TO_TEST if exp in available_exps]

    print(f"Available experiments: {available_exps}")
    print(f"Training on: {train_ids}")
    print(f"Testing on: {test_ids}")

    if not train_ids:
        print("❌ No training experiments available!")
        return

    # Hyperparameters dictionary
    hyperparams = {
        'history_steps': HISTORY_STEPS,
        'augmentation': USE_AUGMENTATION,
        'num_augmented_points': NUM_AUGMENTED_POINTS,
        'model_hidden_width': MODEL_HIDDEN_WIDTH,
        'model_hidden_depth': MODEL_HIDDEN_DEPTH,
        'model_type': MODEL_TYPE
    }

    # Train or load model
    print("\n🧠 Training ensemble model...")
    ensemble = train_cv_ensemble_with_parallel(
        train_ids=train_ids,
        All_Data_Exp=All_Data_Exp,
        max_epochs=MAX_EPOCHS,
        n_jobs=N_JOBS,
        history_steps=HISTORY_STEPS,
        augmentation_flag=USE_AUGMENTATION,
        num_augmented_points=NUM_AUGMENTED_POINTS,
        n_splits=N_SPLITS,
        model_hidden_width=MODEL_HIDDEN_WIDTH,
        model_hidden_depth=MODEL_HIDDEN_DEPTH,
        optimizer_lr=LEARNING_RATE,
        optimizer_type=OPTIMIZER_TYPE,
        ode_method=ODE_METHOD,
        ode_rtol=ODE_RTOL,
        ode_atol=ODE_ATOL,
        lambda_smooth=LAMBDA_SMOOTH,
        max_grad_norm=MAX_GRAD_NORM,
        prefer_min_sum_loss=PREFER_MIN_SUM_LOSS,
        use_reduce_on_plateau=USE_REDUCE_ON_PLATEAU,
        model_type=MODEL_TYPE
    )


    # Save model
    if SAVE_MODEL:
        model_path = os.path.join(OUTPUT_DIR, "trained_ensemble.pth")
        save_ensemble(
                ensemble,
                model_path,
                HISTORY_STEPS,
                MODEL_HIDDEN_WIDTH,
                MODEL_HIDDEN_DEPTH
        )
        print(f"💾 Model saved to {model_path}")

    # Test on unseen experiments
    if test_ids:
        print("\n🧪 Testing on unseen experiments...")

        # Prepare dictionaries for predictions
        Data_Simulated_All = {}
        Predicted_Std_All = {}

        for exp_id in test_ids:
            print(f"\n🔬 Predicting experiment {exp_id}")

            # Get experimental data
            df = All_Data_Exp[exp_id]

            # Prepare time span
            t_span = torch.tensor(df['Time'].values, dtype=torch.float32)

            # Predict with ensemble
            pred_mean, pred_std = predict_with_ensemble(
                models_and_scalers=ensemble,
                df=df,
                t_span=t_span,
                cols=['Biomass', 'Acetate', 'Glucose', 'Glycerol', 'pDNA'],
                exp_id=exp_id,
                history_steps=HISTORY_STEPS,
                ode_method=ODE_METHOD,
                ode_rtol=ODE_RTOL,
                ode_atol=ODE_ATOL,
                n_jobs=N_JOBS
            )

            # Store predictions
            Data_Simulated_All[exp_id] = pd.DataFrame({
                'Time': t_span.numpy(),
                'Biomass': pred_mean[:, 0],
                'Acetate': pred_mean[:, 1],
                'Glucose': pred_mean[:, 2],
                'Glycerol': pred_mean[:, 3],
                'pDNA': pred_mean[:, 4],
                'Volume': pred_mean[:, 5]
            })

            Predicted_Std_All[exp_id] = pred_std[:, 4]  # pDNA std

        # Plot predictions
        print("\n📈 Generating prediction plots...")
        plot_prediction_with_variation(
            Data_Simulated_All=Data_Simulated_All,
            All_Data_Exp=All_Data_Exp,
            Num_Exp=test_ids,
            Predicted_Std_All=Predicted_Std_All,
            output_dir=os.path.join(OUTPUT_DIR, "figures")
        )

        # Calculate and plot MAE
        print("\n📊 Calculating error metrics...")
        df_mae = calculate_and_plot_mae(
            Data_Simulated=Data_Simulated_All,
            All_Data_Exp=All_Data_Exp,
            experiment_ids=test_ids,
            title_prefix="Test_Set",
            output_dir=os.path.join(OUTPUT_DIR, "metrics")
        )

        print("\n📋 MAE Summary:")
        print(df_mae.to_string())

        # Save metrics to CSV
        metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.csv")
        df_mae.to_csv(metrics_path)
        print(f"💾 Metrics saved to {metrics_path}")

    else:
        print("⚠️ No test experiments available!")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print(f"📁 Results saved in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()