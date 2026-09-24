
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
def prepare_morphology_features(uni_panel1_list, uni_panel2_list,
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
    for uni_panel1, uni_panel2 in zip(uni_panel1_list, uni_panel2_list):
        X_panel1 = uni_panel1.X
        X_panel2 = uni_panel2.X
        if issparse(X_panel1): X_panel1 = X_panel1.toarray()
        if issparse(X_panel2):  X_panel2  = X_panel2.toarray()

        m_panel1 = normalize(pca.transform(X_panel1.astype(np.float32)))
        m_panel2  = normalize(pca.transform(X_panel2.astype(np.float32)))

        morph_480_list.append(m_panel1)
        morph_5k_list.append(m_panel2)
        print( f"panel 1 morph {m_panel1.shape}, panel 2 morph {m_panel2.shape}")

    return morph_480_list, morph_5k_list, pca

#normalize and log transform expression data 
def get_normalized_X(adata, genes):
        a = adata[:, genes].copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        X = a.X.toarray() if issparse(a.X) else np.array(a.X)
        return X
    

def prepare_expression_features(rna_panel1, rna_panel2, n_comps=50, normalize_expr = False):

    print("\n" + "=" * 55)
    print("  Preparing expression features (overlapping genes PCA)")
    print("=" * 55)

    # Find genes overlapping across ALL samples
    overlap_genes = set(rna_panel1[0].var_names)
    for sample in rna_panel1[1:] + rna_panel2:
        overlap_genes &= set(sample.var_names)
        
        
    overlap_genes = sorted(overlap_genes)
    print(f"  Overlapping genes across all samples: {len(overlap_genes)}")

    

    
    if normalize_expr:
        
        X_panel1_all = np.vstack([
            get_normalized_X(rna_panel1_sample, overlap_genes)
            for rna_panel1_sample in rna_panel1
        ])
        
    else:
        # :white_check_mark: Extract .X and convert to dense numpy array
        X_panel1_all = np.vstack([
            rna_panel1_sample[:, overlap_genes].X.toarray()
            if issparse(rna_panel1_sample[:, overlap_genes].X)
            else np.array(rna_panel1_sample[:, overlap_genes].X)
            for rna_panel1_sample in rna_panel1
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

    for rna_panel1_sample, rna_panel2_sample in zip(rna_panel1, rna_panel2):
        
        if normalize_expr:
            X_panel1 = get_normalized_X(rna_panel1_sample, overlap_genes)
            X_panel2  = get_normalized_X(rna_panel2_sample,  overlap_genes)
        else:
            X_panel1 = (rna_panel1_sample[:, overlap_genes].X.toarray()
                        if issparse(rna_panel1_sample[:, overlap_genes].X)
                        else np.array(rna_panel1_sample[:, overlap_genes].X))
        
            X_panel2  = (rna_panel2_sample[:, overlap_genes].X.toarray()
                         if issparse(rna_panel2_sample[:, overlap_genes].X)
                         else np.array(rna_panel2_sample[:, overlap_genes].X))
        

        # Transform and normalize each panel's expression
        e_panel1 = normalize(pca.transform(X_panel1))
        e_panel2  = normalize(pca.transform(X_panel2))

        expr_panel1_list.append(e_panel1)
        expr_panel2_list.append(e_panel2)

        print(f"Panel 1 expr {e_panel1.shape}, Panel 2 expr {e_panel1.shape}")


    return expr_panel1_list, expr_panel2_list, pca, overlap_genes
