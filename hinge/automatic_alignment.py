import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path




#Used in Step 0: Automatic allignment



#returns (2,2) matrix, the first vector is the first principal axis that captures the most variance, the second column is perpendicular to the first principal axis
def pca_axes(pts):
    cov = np.cov(pts.T) #calculate covariance matrix
    eigval, eigvec = np.linalg.eigh(cov) # calculate eigenvalues and eigenvectors
    return eigvec[:, np.argsort(eigval)[::-1]] # return eigenvectors sorted by eigenvalues in descending order
    
    
    
#aligns the two points to have the same orientation and position
#can only align by 90 degree rotations and flips, not arbitrary rotations
def coarse_align(source, target):
    
    #center both point clouds
    src_c = source - source.mean(0)
    tgt_c = target - target.mean(0)

    
    #get PCA axes for both point clouds
    R_src = pca_axes(src_c)
    R_tgt = pca_axes(tgt_c)
    
    
    best_result, best_err = None, np.inf

    #test all combinations of flipping the axes (4 combinations)
    for fx in [1, -1]:
        for fy in [1, -1]:
            
            F       = np.diag([fx, fy]) #creates 2x2 diagonal matrix with fx and fy on the diagonal, used to flip the axes
            R       = R_tgt @ F @ R_src.T #calculate the rotation matrix to match the source PCA axes to the target PCA axes
            src_rot = src_c @ R.T #rotate the source points
            tree    = cKDTree(tgt_c) #create a k-d tree for the target points
            dist, _ = tree.query(src_rot) #find the nearest neighbors
            err     = np.median(dist) #calculate the median distance
            
            #saves the best result and error if the current error is less than the best error
            if err < best_err:
                best_err    = err
                best_result = src_rot + target.mean(0)

    print(f"    [coarse_align] best flip median dist: {best_err:.2f}")
    return best_result





# calculate the affine transformation that best aligns the source points to the target points
# based on the Iterative Closest Point (ICP) algorithm
# but with a trimming step to reduce the effect of outliers
def icp_affine_robust(source, target, n_iter=100, tol=1e-7, trim_frac=0.85):
    
    src, tree, prev_err = source.copy(), cKDTree(target), np.inf

    # for each iteration
    for iteration in range(n_iter):
        dist, idx      = tree.query(src)#get closest points in target for each point in source
        matched_target = target[idx] #get the matched target points for each source point
        
        #trims farthest points to reduce the effect of outliers
        n_keep         = int(len(dist) * trim_frac)
        keep_idx       = np.argsort(dist)[:n_keep]
        
        
        src_h          = np.hstack([src, np.ones((len(src), 1))]) #converts each point in src to [x, y, 1] for affine transformation
        
        
        A, *_          = np.linalg.lstsq(
            src_h[keep_idx], matched_target[keep_idx], rcond=None
        ) #calculate the affine transformation matrix A that maps src_h to matched_target using least squares
        
        src = src_h @ A #apply the affine transformation to src_h to get the new src points
        
        
        err = np.median(dist[keep_idx])#calculate the median distance of the inliers
        
        
        if abs(prev_err - err) < tol:
            print(f"    [icp] converged at iteration {iteration+1}, "
                  f"median dist (inliers): {err:.2f}")
            break
        prev_err = err
    else:
        print(f"    [icp] reached max iterations, "
              f"median dist (inliers): {prev_err:.2f}")

    return src, A


#applies an affine transformation to a ST points
def apply_affine(xy, A):
    xy_h = np.hstack([xy, np.ones((len(xy), 1))])
    return xy_h @ A




# a wrapper function
# that aligns two st datasets using coarse and fine alignment
# returns the aligned source points and the affine transformation matrix
def align_section_pair(xy_panel1, xy_panel2, pre_aligned=False,
                        trim_frac=0.85, n_iter=100, tol=1e-7,
                        pair_label=""):
    label = f"[{pair_label}] " if pair_label else ""

    #skip if the user has already pre-aligned the two datasets
    if pre_aligned:
        print(f"{label}pre_aligned=True -> skipping registration")
        return xy_panel2.copy(), None

    print(f"{label}Stage 1/2: Coarse PCA alignment...")
    xy_panel2_coarse = coarse_align(xy_panel2, xy_panel1)

    print(f"{label}Stage 2/2: Robust affine ICP "
          f"(trim_frac={trim_frac}, n_iter={n_iter})...")
    xy_panel2_registered, A = icp_affine_robust(
        xy_panel2_coarse, xy_panel1,
        n_iter=n_iter, tol=tol, trim_frac=trim_frac
    )

    tree = cKDTree(xy_panel1)
    final_dist, _ = tree.query(xy_panel2_registered)
    print(f"{label}Registration quality:")
    print(f"    median dist : {np.median(final_dist):.2f} um")
    print(f"    % < 10 um   : {(final_dist < 10).mean():.1%}")
    print(f"    % < 20 um   : {(final_dist < 20).mean():.1%}")
    print(f"    % > 50 um   : {(final_dist > 50).mean():.1%}  "
          f"<- likely unmatched / tissue boundary")

    return xy_panel2_registered, A




# updates the spatial coordinates of a st dataset 
# for every st pair in the registration_results dictionary
# the updated coordinates are stored in the uni_panel2_list and rna_data dictionaries
def update_coordinates(registration_results, pairs,
                           uni_panel2_list, rna_data):
    print("\n" + "=" * 55)
    print("  Updating spatial coordinates to registered frame")
    print("=" * 55)

    for pair_idx, (name_panel1, name_panel2) in enumerate(pairs):
        result      = registration_results[(name_panel1, name_panel2)]
        A           = result['A']
        xy_reg      = result['xy_panel2_registered']
        pre_aligned = (A is None)

        print(f"\n[{name_panel1} / {name_panel2}]")
        
        
        # if the sample pairs are already pre-aligned using another method,
        # just copy the spatial coordinates from rna_data to uni_panel2_list
        if pre_aligned:
            uni_panel2_list[pair_idx].obsm['spatial'] = rna_data[name_panel2].obsm['spatial'].copy()
            
            rna_data[name_panel2].obsm['spatial_registered'] = rna_data[name_panel2].obsm['spatial'].copy()
            print("  pre_aligned=True, no transformation, stored as 'spatial_registered'")


        else:
            # Update UNI token spatial coordinates of the second panel using the affine transformation matrix A
            orig_uni = uni_panel2_list[pair_idx].obsm['spatial']
            uni_panel2_list[pair_idx].obsm['spatial'] = apply_affine(orig_uni, A)

            # Preserve the original spatial coordinates of the second panel in rna_data
            # and add a new key 'spatial_registered' for the transformed coordinates
            # calculates the transformed coordinates of the second panel using the affine transformation matrix A
            # then checks the maximum difference between the transformed coordinates and the registered coordinates from the registration_results dictionary
            # to make sure the transformation is consistent with the registration results
            orig_rna = rna_data[name_panel2].obsm['spatial']
            reg_rna  = apply_affine(orig_rna, A)
            rna_data[name_panel2].obsm['spatial_registered'] = reg_rna

            print(f"  uni_panel2_list[{pair_idx}].obsm['spatial']              -> updated")
            print(f"  rna_data['{name_panel2}'].obsm['spatial_registered'] -> added")

            max_diff = np.abs(reg_rna - xy_reg).max()
            status   = "PASSED" if max_diff < 1e-4 else f"WARNING max diff={max_diff:.2e}"
            print(f"  Consistency check: {status}")





def plot_alignment_results(pairs, rna_data, registration_results,
                            save_dir, n_sample=3000,
                            panel1_name = "480", panel2_name = "5k"):
    """
    For each section pair, creates a separate 3-panel figure:
        Left   : before alignment
        Middle : after alignment
        Right  : distance distribution after alignment

    Saves one file per pair to {save_dir}/alignment_{name_panel1}_{name_panel2}.png
    """
    for name_panel1, name_panel2 in pairs:
        
        #get the spatial coordinates of the two panels and the registered coordinates of the second panel
        xy_panel1 = rna_data[name_panel1].obsm['spatial']
        xy_panel2  = rna_data[name_panel2].obsm['spatial']
        xy_reg = registration_results[(name_panel1, name_panel2)]['xy_panel2_registered']


        # randomly sample a subset of points for plotting to avoid overcrowding the scatter plot
        rng   = np.random.default_rng(42)
        i_panel1 = rng.choice(len(xy_panel1), min(n_sample, len(xy_panel1)), replace=False)
        i_panel2  = rng.choice(len(xy_panel2),  min(n_sample, len(xy_panel2)),  replace=False)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Alignment: {name_panel1} ({panel1_name} panel)  /  {name_panel2} ({panel2_name} panel)',
                     fontsize=13, fontweight='bold')

        # ── Panel 1: Before ───────────────────────────────────────
        ax = axes[0]
        ax.scatter(*xy_panel1[i_panel1].T, s=1, alpha=0.4, c='royalblue',
                   label=f'{name_panel1} ({panel1_name} panel)')
        ax.scatter(*xy_panel2[i_panel2].T,   s=1, alpha=0.4, c='tomato',
                   label=f'{name_panel2} ({panel2_name} panel), raw)')
        ax.set_title('BEFORE alignment', fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend(markerscale=8, fontsize=8, loc='upper right')

        # ── Panel 2: After ────────────────────────────────────────
        ax = axes[1]
        ax.scatter(*xy_panel1[i_panel1].T, s=1, alpha=0.4, c='royalblue',
                   label=f'{name_panel1} ({panel1_name} panel)')
        ax.scatter(*xy_reg[i_panel2].T,  s=1, alpha=0.4, c='tomato',
                   label=f'{name_panel2} ({panel2_name} panel), registered)')
        ax.set_title('AFTER alignment', fontsize=11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.legend(markerscale=8, fontsize=8, loc='upper right')

        # ── Panel 3: Distance histogram ───────────────────────────
        ax      = axes[2]
        tree    = cKDTree(xy_reg)
        dist, _ = tree.query(xy_panel1)

        ax.hist(dist, bins=80, range=(0, 150),
                color='steelblue', alpha=0.8, edgecolor='none')

        med = np.median(dist)
        ax.axvline(med,  color='red',    linestyle='--', linewidth=1.5,
                   label=f'median = {med:.1f} um')
        ax.axvline(50.0, color='orange', linestyle=':',  linewidth=1.5,
                   label='50 um cutoff')

        p25, p75, p90 = np.percentile(dist, [25, 75, 90])
        ax.text(0.97, 0.95,
                f'p25={p25:.1f}  p75={p75:.1f}  p90={p90:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='dimgray')

        ax.set_xlabel('Nearest-neighbor distance (um)', fontsize=10)
        ax.set_ylabel('Cell count', fontsize=10)
        ax.set_title('Post-alignment distance distribution', fontsize=11)
        ax.legend(fontsize=9)

        plt.tight_layout()

        out = Path(save_dir) / f'alignment_{name_panel1}_{name_panel2}.png'
        plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.show()
        print(f"Saved: {out}")