# kernel.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from itertools import combinations
from torchdiffeq import odeint
import random

# ==============================================================================
# GLOBAL CONFIGURATION & STYLING
# ==============================================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Standard plot settings matching the original style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})
sns.set_theme(style="whitegrid")
dpi = 300  # Adjusted for standard screens
EXCEL_FILE = "Dataset.xlsx"

# ==============================================================================
# METADATA MAPS & ONE-HOT ENCODING (Strictly Exps 1-5)
# ==============================================================================

strain_map = {
    '1': 'DH5a', '2': 'DH5a', '3': 'DH5a', '4': 'DH5a', '5': 'DH5a'
}
strain_to_index = {'DH5a': 0}

pDNA_map = {
    '1': 'pVAX-LacZ', '2': 'pVAX-LacZ', '3': 'pVAX-LacZ', '4': 'pVAX-LacZ', '5': 'pVAX-LacZ'
}
pDNA_to_index = {'pVAX-LacZ': 0}

media_map = {
    '1': 'LB', '2': 'LB', '3': 'LB', '4': 'LB', '5': 'LB'
}
media_to_index = {'LB': 0}

pH_map = {
    '1': 'seven', '2': 'seven', '3': 'seven', '4': 'seven', '5': 'seven'
}
pH_to_index = {'seven': 0}

carbon_source_map = {
    '1': 'glucose', '2': 'glycerol', '3': 'glucose_glycerol',
    '4': 'glucose_glycerol', '5': 'glucose_glycerol'
}
carbon_source_to_index = {'glucose': 0, 'glycerol': 1, 'glucose_glycerol': 2}

one_hot_dim = (
        len(strain_to_index) + len(pDNA_to_index) + len(media_to_index) +
        len(pH_to_index) + len(carbon_source_to_index)
)


def get_one_hot(exp_id):
    """Convert experiment ID to concatenated one-hot vector."""

    def get_vec(val, idx_map):
        vec = np.zeros(len(idx_map))
        if val in idx_map:
            vec[idx_map[val]] = 1.0
        return vec

    return np.concatenate([
        get_vec(strain_map.get(exp_id), strain_to_index),
        get_vec(pDNA_map.get(exp_id), pDNA_to_index),
        get_vec(media_map.get(exp_id), media_to_index),
        get_vec(pH_map.get(exp_id), pH_to_index),
        get_vec(carbon_source_map.get(exp_id), carbon_source_to_index)
    ])


# ==============================================================================
# DATA PREPARATION
# ==============================================================================

def replace_nans(series, mode='interp'):
    if mode == 'zero':
        return np.nan_to_num(series, nan=0.0)
    elif mode == 'interp':
        return series.interpolate(method='linear', limit_direction='both')
    return series


def augment_experimental_data_np(raw_target, t_exp, cols, num_augmented_points=100):
    from scipy.interpolate import make_interp_spline
    t_new = np.linspace(t_exp[0], t_exp[-1], num_augmented_points)
    data_aug = {'Time': t_new}

    for i, col in enumerate(cols):
        y = raw_target[:, i]
        if len(t_exp) < 4:
            # Linear interpolation for few points
            y_spline = np.interp(t_new, t_exp, y)
        else:
            # Cubic spline for sufficient points
            try:
                spline = make_interp_spline(t_exp, y, k=min(3, len(t_exp) - 1))
                y_spline = spline(t_new)
            except:
                y_spline = np.interp(t_new, t_exp, y)
        data_aug[col] = y_spline
    return data_aug


def prepare_data(df, exp_id=None, do_augmentation=False, num_augmented_points=100):
    df = df.copy()
    cols = ['Biomass', 'Acetate', 'Glucose', 'Glycerol', 'pDNA', 'Volume']

    # Fill NaNs
    for col in cols[:-1]:
        df[col] = replace_nans(df[col], mode='interp')

    t_exp = df['Time'].values.astype(np.float32)

    if do_augmentation:
        aug = augment_experimental_data_np(df[cols].values, t_exp, cols, num_augmented_points)
        t_span = torch.tensor(aug['Time'], dtype=torch.float32)
        target = torch.tensor(np.stack([aug[k] for k in cols], 1), dtype=torch.float32)
    else:
        t_span = torch.tensor(t_exp, dtype=torch.float32)
        target = torch.tensor(df[cols].values, dtype=torch.float32)

    init = torch.tensor([df[c].iloc[0] for c in cols], dtype=torch.float32)

    one_hot = torch.zeros(one_hot_dim, dtype=torch.float32)
    if exp_id is not None:
        one_hot = torch.tensor(get_one_hot(exp_id), dtype=torch.float32)

    return t_span, target, init, one_hot


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class RateNetwork(nn.Module):
    """Feed-Forward Network for kinetics."""

    def __init__(self, input_dim, width=64, depth=8):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.LayerNorm(width))
        layers.append(nn.ReLU())
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_rate_model(history_steps=3, width=64, depth=8, model_type='fnn'):
    """Factory to create the hybrid rate model."""
    input_dim = 5 + 5 * history_steps + one_hot_dim

    # We create a dictionary of networks, one for each specific kinetic rate
    return nn.ModuleDict({
        'r_X': RateNetwork(input_dim, width, depth),
        'r_A': RateNetwork(input_dim, width, depth),
        'r_S': RateNetwork(input_dim, width, depth),
        'r_G': RateNetwork(input_dim, width, depth),
        'r_P': RateNetwork(input_dim, width, depth),
        'r_PA': RateNetwork(input_dim, width, depth)
    })


# ==============================================================================
# ODE SYSTEMS
# ==============================================================================

class EcoliODE(nn.Module):
    """Batch fermentation ODE system."""

    def __init__(self, rate_model, history_steps=3, one_hot_vector=None, t_f=1.0):
        super().__init__()
        self.rate_model = rate_model
        self.history_steps = history_steps
        self.register_buffer('dx_history', torch.zeros(history_steps, 5))
        self.register_buffer('one_hot_vector',
                             (one_hot_vector if one_hot_vector is not None else torch.zeros(one_hot_dim)).unsqueeze(0))

    def reset_history(self):
        self.dx_history.zero_()

    def forward(self, t, x):
        X, A, S, G, P, V = x.unbind(0)

        # Prepare inputs: Current state + history + static metadata
        rate_inputs = torch.cat([
            x[:5].clamp(min=0),
            self.dx_history.flatten(),
            self.one_hot_vector.flatten()
        ])

        rates = {k: net(rate_inputs) for k, net in self.rate_model.items()}

        # Mass Balances (Batch process: volume is constant, no feed/dilution)
        dx = torch.zeros_like(x)
        dx[0] = rates['r_X'] * X
        dx[1] = (-rates['r_A'] + rates['r_PA']) * X
        dx[2] = -rates['r_S'] * X
        dx[3] = -rates['r_G'] * X
        dx[4] = rates['r_P'] * X
        dx[5] = torch.tensor(0.0) # Volume is constant

        # Soft clamping to prevent explosions
        if torch.any(torch.isnan(dx)) or torch.any(torch.isinf(dx)):
            dx = torch.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)

        # Update history
        self.dx_history = torch.cat([self.dx_history[1:], dx[:5].unsqueeze(0).detach()])

        return dx


# ==============================================================================
# TRAINING LOGIC
# ==============================================================================

def mixed_loss(pred, target, t=None, lambda_smooth=1e-6):
    """Relative RMSE."""
    mse = torch.mean((pred - target) ** 2)
    rmse = torch.sqrt(mse)
    # Add epsilon to std to avoid division by zero
    rrmse = rmse / (torch.std(target) + 1e-6)
    return rrmse


def train_single_fold(
        val_ids, train_ids, All_Data_Exp, max_epochs,
        model_hidden_width=32, model_hidden_depth=4,
        optimizer_lr=1e-2, optimizer_type='adam',
        history_steps=3, ode_method='euler', ode_rtol=1e-4, ode_atol=1e-5,
        lambda_smooth=1e-3, max_grad_norm=1.0,
        use_reduce_on_plateau=True, prefer_min_sum_loss=False,
        augmentation_flag=False, num_augmented_points=100, model_type='fnn', **kwargs
):
    """Train a single model on a split of data."""

    # 1. Prepare Data
    train_data = [prepare_data(All_Data_Exp[eid], eid, augmentation_flag, num_augmented_points) for eid in train_ids]
    val_data = [prepare_data(All_Data_Exp[eid], eid, augmentation_flag, num_augmented_points) for eid in
                val_ids] if val_ids else []

    # 2. Init Model
    model = get_rate_model(history_steps, model_hidden_width, model_hidden_depth, model_type)

    # 3. Optimizer & Scheduler
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    # 4. Helper to create ODE instances
    train_odes = [EcoliODE(model, history_steps, oh, t[-1].item()) for (t, _, _, oh), eid in zip(train_data, train_ids)]
    val_odes = [EcoliODE(model, history_steps, oh, t[-1].item()) for (t, _, _, oh), eid in zip(val_data, val_ids)]

    # 5. Loop
    best_loss = float('inf')
    # Initialize with starting weights as fallback in case of NaNs
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        losses = []

        for (t, y, init, _), ode in zip(train_data, train_odes):
            ode.reset_history()
            pred = odeint(ode, init, t, method=ode_method, rtol=ode_rtol, atol=ode_atol).clamp(min=0)
            losses.append(mixed_loss(pred, y, t, lambda_smooth))

        train_loss = torch.stack(losses).mean()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Validation
        val_loss = train_loss  # Default if no val set
        if val_data:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for (t, y, init, _), ode in zip(val_data, val_odes):
                    ode.reset_history()
                    pred = odeint(ode, init, t, method=ode_method, rtol=ode_rtol, atol=ode_atol).clamp(min=0)
                    val_losses.append(mixed_loss(pred, y, t, lambda_smooth))
            val_loss = torch.stack(val_losses).mean()

        if use_reduce_on_plateau:
            scheduler.step(val_loss)

        # Save Best
        metric = val_loss if not prefer_min_sum_loss else (val_loss + train_loss)
        if metric < best_loss:
            best_loss = metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_loss


def train_cv_ensemble_with_parallel(
        train_ids, All_Data_Exp, n_jobs=-1, n_splits=None,
        test_size=0.3, prune_ratio=0.0, **kwargs
):
    """Manager function to run cross-validation training in parallel."""

    # Generate splits
    n_total = len(train_ids)
    if n_total == 1:
        print(f"Single experiment training on {train_ids[0]}")
        model, _ = train_single_fold([], train_ids, All_Data_Exp, **kwargs)
        return [model]

    k = max(1, int(round(n_total * test_size)))
    combos = list(combinations(train_ids, k))
    if n_splits and len(combos) > n_splits:
        random.shuffle(combos)
        combos = combos[:n_splits]

    print(f"Training ensemble with {len(combos)} folds on {n_jobs} cores...")

    tasks = []
    for val_set in combos:
        val_list = list(val_set)
        train_list = [x for x in train_ids if x not in val_set]
        tasks.append(delayed(train_single_fold)(val_list, train_list, All_Data_Exp, **kwargs))

    results = Parallel(n_jobs=n_jobs)(tasks)

    # Pruning based on validation loss
    models = [r[0] for r in results]
    losses = [r[1].item() if isinstance(r[1], torch.Tensor) else r[1] for r in results]

    sorted_idx = np.argsort(losses)
    n_keep = max(1, int(len(models) * (1 - prune_ratio)))
    best_models = [models[i] for i in sorted_idx[:n_keep]]

    print(f"Ensemble training complete. Kept {len(best_models)} models.")
    return best_models


# ==============================================================================
# PREDICTION LOGIC
# ==============================================================================

def run_single_prediction(model, df, t_span, exp_id, history_steps, ode_method, ode_rtol, ode_atol):
    """Worker for prediction."""
    t_span, _, init, one_hot = prepare_data(df, exp_id, do_augmentation=False)

    ode = EcoliODE(model, history_steps, one_hot, t_span[-1].item())

    with torch.no_grad():
        ode.reset_history()
        pred = odeint(ode, init, t_span, method=ode_method, rtol=ode_rtol, atol=ode_atol).clamp(min=0)

    return pred.numpy()


def predict_with_ensemble(
        models_and_scalers, df, t_span, cols, exp_id,
        history_steps=3, ode_method='euler', ode_rtol=1e-4, ode_atol=1e-5, n_jobs=-1, **kwargs
):
    """Predict using ensemble mean and std."""

    tasks = [delayed(run_single_prediction)(
        m, df, t_span, exp_id, history_steps, ode_method, ode_rtol, ode_atol
    ) for m in models_and_scalers]

    preds = Parallel(n_jobs=n_jobs)(tasks)
    stack = np.stack(preds, axis=0)

    return stack.mean(axis=0), stack.std(axis=0)


# ==============================================================================
# PERSISTENCE & PLOTTING
# ==============================================================================

def save_ensemble(models, filepath, history_steps, width, depth):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'model_states': [m.state_dict() for m in models],
        'history_steps': history_steps,
        'model_hidden_width': width,
        'model_hidden_depth': depth
    }, filepath)


def load_ensemble(filepath):
    checkpoint = torch.load(filepath)
    models = []
    for state in checkpoint['model_states']:
        m = get_rate_model(checkpoint['history_steps'], checkpoint['model_hidden_width'],
                           checkpoint['model_hidden_depth'])
        m.load_state_dict(state)
        models.append(m)
    return models


def plot_prediction_with_variation(Data_Simulated_All, All_Data_Exp, Num_Exp, Predicted_Std_All, output_dir="Figures"):
    os.makedirs(output_dir, exist_ok=True)
    vars = ['Biomass', 'Glucose', 'Glycerol', 'Acetate', 'pDNA']
    colors = sns.color_palette('deep', len(vars))

    for exp_id in Num_Exp:
        sim = Data_Simulated_All[exp_id]
        raw = All_Data_Exp[exp_id]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Plot standard vars
        for i, var in enumerate(vars[:-1]):
            if var in sim.columns:
                ax1.plot(sim['Time'], sim[var], color=colors[i], label=f'{var} (Sim)')
                if var in raw.columns:
                    ax1.scatter(raw['Time'], raw[var], color=colors[i], marker='o')

        # Plot pDNA on secondary axis
        p_var = 'pDNA'
        if p_var in sim.columns:
            ax2.plot(sim['Time'], sim[p_var], color='red', linestyle='-', label='pDNA (Sim)')
            if p_var in raw.columns:
                ax2.scatter(raw['Time'], raw[p_var], color='red', marker='o')

            # Std deviation shading
            if exp_id in Predicted_Std_All:
                std = Predicted_Std_All[exp_id]
                ax2.fill_between(sim['Time'], sim[p_var] - std, sim[p_var] + std, color='red', alpha=0.1)

        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Conc (g/L)')
        ax2.set_ylabel('pDNA (mg/L)')
        fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        plt.title(f'Experiment {exp_id}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Exp_{exp_id}.png'), dpi=dpi)
        plt.close()


def calculate_and_plot_mae(Data_Simulated, All_Data_Exp, experiment_ids, output_dir="Metrics", **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    mae_results = {}

    for exp in experiment_ids:
        sim = Data_Simulated[exp]
        raw = All_Data_Exp[exp]
        maes = {}
        for col in ['Biomass', 'Glucose', 'pDNA']:
            if col in raw and col in sim:
                # Interp sim to raw timepoints
                clean_raw = raw.dropna(subset=[col])
                if len(clean_raw) == 0: continue
                sim_vals = np.interp(clean_raw['Time'], sim['Time'], sim[col])
                maes[col] = np.mean(np.abs(sim_vals - clean_raw[col].values))
        mae_results[exp] = maes

    df = pd.DataFrame(mae_results).T

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('MAE by Experiment and Variable')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'MAE_Heatmap.png'), dpi=dpi)
    plt.close()

    return df