from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_neighborhood_consistency(
    xy_panel1, xy_panel2_reg,
    idx_panel1, idx_panel2,
    adata_panel1, adata_panel2,
    k_neighbors  = 15,
    celltype_col = 'cell_type'
):
    #check if celltype column is available in both adata_panel1 and adata_panel2
    assert celltype_col in adata_panel1.obs.columns, \
        f"'{celltype_col}' not in adata_panel1.obs — available: " \
        f"{adata_panel1.obs.columns.tolist()}"
    assert celltype_col in adata_panel2.obs.columns, \
        f"'{celltype_col}' not in adata_panel2.obs"


    #get the cell types for each panel
    celltype_panel1    = adata_panel1.obs[celltype_col].values
    celltype_panel2     = adata_panel2.obs[celltype_col].values
    
    #get list of cell types from both panels
    all_types = sorted(set(celltype_panel1) | set(celltype_panel2))
    
    #create a dictionary mapping cell types to indices
    ct_idx    = {ct: i for i, ct in enumerate(all_types)}
    
    n_celltypes   = len(all_types)

    tree_panel1 = cKDTree(xy_panel1)
    tree_panel2  = cKDTree(xy_panel2_reg)

    scores = []
    
    #for each cell pairs
    #get the k nearest neighbors for each cell in both panels
    #get the cell type distribution of the neighbors
    #compute the cosine similarity between the two distributions
    for i, j in zip(idx_panel1, idx_panel2):
        _, n_panel1 = tree_panel1.query(xy_panel1[i],    k=k_neighbors + 1)
        _, n_panel2  = tree_panel2.query(xy_panel2_reg[j],  k=k_neighbors + 1)
        n_panel1, n_panel2 = n_panel1[1:], n_panel2[1:]   # exclude self

        v_panel1 = np.zeros(n_celltypes)
        v_panel2  = np.zeros(n_celltypes)
        for n in n_panel1: v_panel1[ct_idx[celltype_panel1[n]]] += 1
        for n in n_panel2:  v_panel2[ct_idx[celltype_panel2[n]]]  += 1

        norm_v_panel1 = np.linalg.norm(v_panel1)
        norm_v_panel2  = np.linalg.norm(v_panel2)
        if norm_v_panel1 > 0 and norm_v_panel2 > 0:
            scores.append(np.dot(v_panel1, v_panel2) / (norm_v_panel1 * norm_v_panel2))

    return np.array(scores)



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



STRATEGY_LABELS = {
    'A_distance'  : 'Distance\nonly',
    'B_histology' : 'Histology\nonly',
    'C_expression': 'Expression\nonly',
    'D_combined'  : 'Histology +\nExpression',
}
STRATEGY_COLORS = {
    'A_distance'  : '#95a5a6',
    'B_histology' : '#3498db',
    'C_expression': '#2ecc71',
    'D_combined'  : '#e74c3c',
}





def plot_ablation_summary(all_eval_scores):
    """
    3-panel summary figure pooled across all 6 pairs:
        Left   : violin plot of score distributions
        Middle : median bar chart with delta vs baseline
        Right  : cumulative distribution
    """
    strategies = list(STRATEGY_LABELS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Matching Strategy Ablation — Neighborhood Consistency\n'
        '(pooled across all 6 section pairs)',
        fontsize=13, fontweight='bold'
    )

    # ── Violin ───────────────────────────────────────────────────
    ax    = axes[0]
    data  = [all_eval_scores[s] for s in strategies]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc, s in zip(parts['bodies'], strategies):
        pc.set_facecolor(STRATEGY_COLORS[s])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(strategies) + 1))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=9)
    ax.set_ylabel('Neighborhood Consistency Score')
    ax.set_title('Score Distributions')
    ax.set_ylim(0, 1)
    ax.axhline(np.median(all_eval_scores['A_distance']),
               color='gray', linestyle='--', alpha=0.5,
               label='Distance baseline')
    ax.legend(fontsize=8)

    # ── Bar chart ────────────────────────────────────────────────
    ax      = axes[1]
    medians = [np.median(all_eval_scores[s]) for s in strategies]
    ses     = [np.std(all_eval_scores[s]) /
               np.sqrt(len(all_eval_scores[s])) for s in strategies]
    bars    = ax.bar(range(len(strategies)), medians,
                     color=[STRATEGY_COLORS[s] for s in strategies],
                     alpha=0.8, yerr=ses, capsize=5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=9)
    ax.set_ylabel('Median Consistency Score')
    ax.set_title('Median ± SE')
    baseline = medians[0]
    for bar, med, se in zip(bars, medians, ses):
        delta = med - baseline
        color = 'green' if delta > 0 else 'red'
        sign  = '+' if delta >= 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2,
                med + se + 0.003,
                f'{sign}{delta:.3f}',
                ha='center', fontsize=8,
                color=color, fontweight='bold')

    # ── CDF ──────────────────────────────────────────────────────
    ax = axes[2]
    for s in strategies:
        arr = np.sort(all_eval_scores[s])
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf,
                label=STRATEGY_LABELS[s].replace('\n', ' '),
                color=STRATEGY_COLORS[s], linewidth=2)
    ax.set_xlabel('Consistency Score')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    #out = Path(save_dir) / 'ablation_summary.png'
    #plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.show()
    #print(f"Saved: {out}")


def plot_ablation_per_pair(per_pair_scores):
    """
    One figure per pair showing score distributions across strategies.
    Lets you see which pairs benefit most from biological signals.
    """
    strategies = list(STRATEGY_LABELS.keys())

    for scores in per_pair_scores.items():

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Neighborhood Consistency',
                     fontsize=12, fontweight='bold')

        # Violin
        ax    = axes[0]
        data  = [scores[s] for s in strategies]
        parts = ax.violinplot(data, showmedians=True)
        for pc, s in zip(parts['bodies'], strategies):
            pc.set_facecolor(STRATEGY_COLORS[s]); pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(strategies) + 1))
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies],
                           fontsize=9)
        ax.set_ylabel('Consistency Score')
        ax.set_ylim(0, 1)
        ax.set_title('Score distribution')

        # Bar
        ax      = axes[1]
        medians = [np.median(scores[s]) for s in strategies]
        bars    = ax.bar(range(len(strategies)), medians,
                         color=[STRATEGY_COLORS[s] for s in strategies],
                         alpha=0.8)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies],
                           fontsize=9)
        ax.set_ylabel('Median Consistency Score')
        ax.set_title('Median per strategy')
        baseline = medians[0]
        for bar, med in zip(bars, medians):
            delta = med - baseline
            color = 'green' if delta > 0 else 'red'
            sign  = '+' if delta >= 0 else ''
            ax.text(bar.get_x() + bar.get_width() / 2,
                    med + 0.003,
                    f'{sign}{delta:.3f}',
                    ha='center', fontsize=8,
                    color=color, fontweight='bold')

        plt.tight_layout()

        plt.show()
        #print(f"Saved: {out}")

def print_summary_table(all_eval_scores, per_pair_scores, pairs):
    """Print a clean console summary table."""
    strategies = list(STRATEGY_LABELS.keys())

    print("\n" + "=" * 70)
    print("  POOLED RESULTS (all pairs)")
    print("=" * 70)
    print(f"  {'Strategy':<25} {'Median':>8} {'Mean':>8} "
          f"{'vs baseline':>12} {'n':>8}")
    print("-" * 70)
    baseline_med = np.median(all_eval_scores['A_distance'])
    for s in strategies:
        arr   = all_eval_scores[s]
        delta = np.median(arr) - baseline_med
        sign  = '+' if delta >= 0 else ''
        print(f"  {STRATEGY_LABELS[s].replace(chr(10),' '):<25} "
              f"{np.median(arr):>8.4f} "
              f"{np.mean(arr):>8.4f} "
              f"{sign+f'{delta:.4f}':>12} "
              f"{len(arr):>8}")

    print("\n" + "=" * 70)
    print("  PER-PAIR MEDIANS")
    print("=" * 70)
    header = f"  {'Pair':<12}" + "".join(
        f"{STRATEGY_LABELS[s].replace(chr(10),' '):>18}" for s in strategies
    )
    print(header)
    print("-" * 70)
    for name_480, name_5k in pairs:
        key  = f"{name_480}/{name_5k}"
        row  = f"  {key:<12}"
        for s in strategies:
            row += f"{np.median(per_pair_scores[key][s]):>18.4f}"
        print(row)
    print("=" * 70)
