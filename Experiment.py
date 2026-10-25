import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="run the experiment thingy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('h5_file', type=str, help="HDF5 data file (audit_data.h5).")
    parser.add_argument('--output_dir', type=str, default="audit_results", help="dir to save analysis plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running Audit on {args.h5_file}")

    try:
        df = pd.read_hdf(args.h5_file, key='audit_data')
    except Exception as e:
        print(f"cvouldnt read h5, error: {e}", file=sys.stderr)
        sys.exit(1)

    factor_cols = [f'factor_{i+1}' for i in range(8)]
    proxy_cols = ['dim_material', 'dim_mobility', 'dim_king_freedom', 'dim_pawn_weakness', 'dim_game_phase']
    
    y_pc = df['y_pc']
    X_factors = df[factor_cols]
    X_proxies = df[proxy_cols]

    print("R^2 test")
    X_train, X_test, y_train, y_test = train_test_split(X_factors, y_pc, test_size=0.2, random_state=42)
    
    probe_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=200, random_state=42, early_stopping=True)
    probe_model.fit(X_train, y_train)
    r2 = probe_model.score(X_test, y_test)
    
    print(f"MLP Probe R² Score: {r2:.6f}")
    if r2 > 0.98:
        print("passed!\n")
    else:
        print("you suck at coding\n")

    # ==========================================================================
    # --- 3.2 Analysis B: Structural Independence (Orthogonality Test) ---
    # ==========================================================================
    print("Orthogonality test")
    corr_matrix_B = X_factors.corr()
    mask = np.eye(len(corr_matrix_B), dtype=bool)
    max_off_diag_corr = corr_matrix_B.where(~mask).abs().max().max()

    print(f"Maximum absolute off-diagonal correlation: {max_off_diag_corr:.4f}")
    if max_off_diag_corr < 0.2:
        print("passed!\n")
    else:
        print("you suck at coding\n")
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix_B, annot=True, cmap='vlag', fmt='.2f', vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plot_path_b = os.path.join(args.output_dir, "orthogonality.png")
    plt.savefig(plot_path_b)
    print(f"Saved visualization to {plot_path_b}")

    print("\nInterpretability grounding")
    
    corr_matrix_C = pd.DataFrame(index=X_proxies.columns, columns=X_factors.columns)
    for proxy_col in X_proxies.columns:
        for factor_col in X_factors.columns:
            corr_matrix_C.loc[proxy_col, factor_col] = X_proxies[proxy_col].corr(X_factors[factor_col])
    corr_matrix_C = corr_matrix_C.astype(float)
    
    print("Correlation matrix (proxy to factor):")
    print(corr_matrix_C.round(4))
    
    success_count = 0
    grounded_proxies = []
    for proxy in corr_matrix_C.index:
        row = corr_matrix_C.loc[proxy].abs()
        if row.max() > 0.5:
            strongest_corr_factor = row.idxmax()
            other_factors = row.drop(strongest_corr_factor)
            if other_factors.max() < 0.3:
                success_count += 1
                grounded_proxies.append(f"'{proxy}' -> '{strongest_corr_factor}' (ρ={corr_matrix_C.loc[proxy, strongest_corr_factor]:.2f})")
    
    print(f"\nFound {success_count} human proxies that strongly and uniquely correlate with a single factor.")
    if grounded_proxies:
        print("Grounded relationships found:", ", ".join(grounded_proxies))

    if success_count >= 2:
        print("pass\n")
    else:
        print("fail\n")

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix_C, annot=True, cmap='vlag', fmt='.2f', vmin=-1, vmax=1)
    plt.title("Analysis C: Proxy-Factor Correlation Matrix (Interpretability Grounding)")
    plt.tight_layout()
    plot_path_c = os.path.join(args.output_dir, "analysis_c_interpretability_grounding.png")
    plt.savefig(plot_path_c)
    print(f"Saved visualization to {plot_path_c}")
    

if __name__ == "__main__":
    main()