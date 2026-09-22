
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.sparse import issparse
import scanpy as sc






"""
    PCA-compress and L2-normalize UNI embeddings for each pair.

    Fits one PCA jointly on ALL samples (480 + 5k combined) so both
    panels live in the same morphological feature space.

    Returns two lists of L2-normalized arrays aligned to `pairs`.
"""
def prepare_morphology_features(uni_panel1_list, uni_panel2_list, pairs,
                                  n_pca_comps=100):

    print("=" * 55)
    print("  Preparing morphology features (UNI embeddings)")
    print("=" * 55)

    # Fit PCA on all samples combined
    print("combining UNI features from both panels...")
    all_X = []
    for adata in uni_panel1_list + uni_panel2_list:
        X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
        all_X.append(X.astype(np.float32))
    all_X = np.vstack(all_X)



    n_comps_actual = min(n_pca_comps, all_X.shape[1] - 1) #get the number of components, either n_pca_comps or the number of features - 1, whichever is smaller
    
    # Fit PCA on all samples combined
    print(f"  Fitting PCA on {all_X.shape} UNI features "
          f"→ {n_comps_actual} components...")
    pca = PCA(n_components=n_comps_actual, random_state=42)
    pca.fit(all_X)
    print(f"  Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    del all_X

    morph_480_list = []
    morph_5k_list  = []

    # Transform and normalize each panel's UNI features
    for pair_idx, (name_panel1, name_panel2) in enumerate(pairs):
        X_panel1 = uni_panel1_list[pair_idx].X
        X_panel2 = uni_panel2_list[pair_idx].X
        if issparse(X_panel1): X_panel1 = X_panel1.toarray()
        if issparse(X_panel2):  X_panel2  = X_panel2.toarray()

        m_panel1 = normalize(pca.transform(X_panel1.astype(np.float32)))
        m_panel2  = normalize(pca.transform(X_panel2.astype(np.float32)))

        morph_480_list.append(m_panel1)
        morph_5k_list.append(m_panel2)
        print(f"  {name_panel1}/{name_panel2}: "
              f"panel 1 morph {m_panel1.shape}, panel 2 morph {m_panel2.shape}")

    return morph_480_list, morph_5k_list, pca

#normalize and log transform expression data 
def get_normalized_X(adata, genes):
        a = adata[:, genes].copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        X = a.X.toarray() if issparse(a.X) else np.array(a.X)
        return X
    

def prepare_expression_features(predicted_rna, pairs, n_comps=50, normalize_expr = False):

    print("\n" + "=" * 55)
    print("  Preparing expression features (overlapping genes PCA)")
    print("=" * 55)

    name_panel1_list = [p[0] for p in pairs]
    name_panel2_list  = [p[1] for p in pairs]

    # Find genes overlapping across ALL samples
    overlap_genes = set(predicted_rna[name_panel1_list[0]].var_names)
    for name in name_panel1_list[1:] + name_panel2_list:
        overlap_genes &= set(predicted_rna[name].var_names)
    overlap_genes = sorted(overlap_genes)
    print(f"  Overlapping genes across all samples: {len(overlap_genes)}")

    

    
    if normalize_expr:
        
        X_panel1_all = np.vstack([
            get_normalized_X(predicted_rna[n], overlap_genes)
            for n in name_panel1_list
        ])
        
    else:
        X_panel1_all = np.vstack([
            predicted_rna[n][:, overlap_genes].copy()
            for n in name_panel1_list
        ])
        
    # Fit PCA on all of panel 1 samples combined
    n_comps_actual = min(n_comps, len(overlap_genes) - 1)
    print(f"  Fitting expression PCA on {X_panel1_all.shape} "
          f"→ {n_comps_actual} components...")
    pca = PCA(n_components=n_comps_actual, random_state=42)
    pca.fit(X_panel1_all)
    print(f"  Variance retained: {pca.explained_variance_ratio_.sum():.1%}")
    del X_panel1_all

    expr_panel1_list = []
    expr_panel2_list  = []

    for name_panel1, name_panel2 in pairs:
        
        if normalize_expr:
            X_480 = get_normalized_X(predicted_rna[name_panel1], overlap_genes)
            X_5k  = get_normalized_X(predicted_rna[name_panel2],  overlap_genes)
        else:
            X_480 = predicted_rna[name_panel1][:, overlap_genes].copy()
            X_5k  = predicted_rna[name_panel2][:, overlap_genes].copy()

            
        

        # Transform and normalize each panel's expression
        e_480 = normalize(pca.transform(X_480))
        e_5k  = normalize(pca.transform(X_5k))

        expr_panel1_list.append(e_480)
        expr_panel2_list.append(e_5k)
        print(f"  {name_panel1}/{name_panel2}: "
              f"panel 1 expr {e_480.shape}, panel 2 expr {e_5k.shape}")

    return expr_panel1_list, expr_panel2_list, pca, overlap_genes
