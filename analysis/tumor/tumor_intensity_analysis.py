"""
Tumor Boundary Intensity Analysis for MRI Segmentation
Analyzes intensity profiles, tumor characteristics, and dataset variability
to inform augmentation strategies for rectal cancer segmentation.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 14})

class TumorIntensityAnalyzer:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.results = []
        
    def get_matched_files(self):
        """Get matched image and label file pairs"""
        image_files = sorted(list(self.images_dir.glob("*.nii.gz")) + 
                           list(self.images_dir.glob("*.nii")))
        label_files = sorted(list(self.labels_dir.glob("*.nii.gz")) + 
                           list(self.labels_dir.glob("*.nii")))
        
        # Match by filename
        matched = []
        for img_file in image_files:
            img_name = img_file.stem.replace('.nii', '')
            for lbl_file in label_files:
                lbl_name = lbl_file.stem.replace('.nii', '')
                if img_name == lbl_name or img_name in lbl_name or lbl_name in img_name:
                    matched.append((img_file, lbl_file))
                    break
        
        return matched
    
    def extract_boundary_region(self, mask, distance_mm=10, spacing=(1.0, 1.0, 1.0)):
        """
        Extract boundary region around tumor
        distance_mm: distance in mm from boundary to consider
        """
        # Convert distance to voxels
        distance_voxels = max(1, int(distance_mm / np.mean(spacing)))
        
        # Get tumor boundary using morphological operations
        eroded = ndimage.binary_erosion(mask, iterations=1)
        boundary = mask.astype(int) - eroded.astype(int)
        
        # Dilate boundary to get region around it
        boundary_region = ndimage.binary_dilation(boundary, iterations=distance_voxels)
        
        return boundary, boundary_region
    
    def compute_distance_transform(self, mask):
        """
        Compute signed distance transform
        Positive inside tumor, negative outside
        """
        # Distance from outside
        dist_outside = ndimage.distance_transform_edt(mask)
        # Distance from inside
        dist_inside = ndimage.distance_transform_edt(~mask.astype(bool))
        
        # Signed distance: positive inside, negative outside
        signed_dist = dist_outside.astype(float)
        signed_dist[~mask.astype(bool)] = -dist_inside[~mask.astype(bool)]
        
        return signed_dist
    
    def extract_intensity_profile(self, image, mask, spacing=(1.0, 1.0, 1.0), 
                                  max_distance_mm=15, bin_size_mm=1):
        """
        Extract intensity profile at different distances from boundary
        """
        signed_dist = self.compute_distance_transform(mask)
        
        # Convert to mm
        signed_dist_mm = signed_dist * np.mean(spacing)
        
        # Create bins
        bins = np.arange(-max_distance_mm, max_distance_mm + bin_size_mm, bin_size_mm)
        
        profile_mean = []
        profile_std = []
        profile_q25 = []
        profile_q75 = []
        bin_centers = []
        
        for i in range(len(bins) - 1):
            bin_mask = (signed_dist_mm >= bins[i]) & (signed_dist_mm < bins[i+1])
            if bin_mask.sum() > 0:
                intensities = image[bin_mask]
                profile_mean.append(np.mean(intensities))
                profile_std.append(np.std(intensities))
                profile_q25.append(np.percentile(intensities, 25))
                profile_q75.append(np.percentile(intensities, 75))
                bin_centers.append((bins[i] + bins[i+1]) / 2)
        
        return {
            'distance': np.array(bin_centers),
            'mean': np.array(profile_mean),
            'std': np.array(profile_std),
            'q25': np.array(profile_q25),
            'q75': np.array(profile_q75)
        }
    
    def compute_tumor_metrics(self, image, mask, spacing=(1.0, 1.0, 1.0)):
        """
        Compute comprehensive tumor intensity metrics
        """
        if mask.sum() == 0:
            return None
        
        # Get tumor and background intensities
        tumor_intensities = image[mask > 0]
        background_mask = (mask == 0) & (image > 0)  # Exclude zeros
        background_intensities = image[background_mask]
        
        # Get boundary
        boundary, boundary_region = self.extract_boundary_region(mask, distance_mm=5, spacing=spacing)
        boundary_intensities = image[boundary > 0]
        
        # Compute gradient magnitude at boundary
        gradient = np.gradient(image.astype(float))
        gradient_magnitude = np.sqrt(sum([g**2 for g in gradient]))
        boundary_gradient = gradient_magnitude[boundary > 0]
        
        metrics = {
            # Basic tumor stats
            'tumor_mean': np.mean(tumor_intensities),
            'tumor_std': np.std(tumor_intensities),
            'tumor_median': np.median(tumor_intensities),
            'tumor_cv': np.std(tumor_intensities) / (np.mean(tumor_intensities) + 1e-8),  # Coefficient of variation
            'tumor_volume_voxels': mask.sum(),
            'tumor_volume_mm3': mask.sum() * np.prod(spacing),
            
            # Background stats
            'background_mean': np.mean(background_intensities) if len(background_intensities) > 0 else 0,
            'background_std': np.std(background_intensities) if len(background_intensities) > 0 else 0,
            
            # Boundary stats
            'boundary_mean': np.mean(boundary_intensities) if len(boundary_intensities) > 0 else 0,
            'boundary_std': np.std(boundary_intensities) if len(boundary_intensities) > 0 else 0,
            'boundary_gradient_mean': np.mean(boundary_gradient) if len(boundary_gradient) > 0 else 0,
            'boundary_gradient_std': np.std(boundary_gradient) if len(boundary_gradient) > 0 else 0,
            
            # Contrast metrics
            'tumor_to_background_contrast': (np.mean(tumor_intensities) - np.mean(background_intensities)) / (np.mean(background_intensities) + 1e-8) if len(background_intensities) > 0 else 0,
            'tumor_to_background_snr': (np.mean(tumor_intensities) - np.mean(background_intensities)) / (np.std(background_intensities) + 1e-8) if len(background_intensities) > 0 else 0,
            
            # Intensity distribution
            'tumor_q25': np.percentile(tumor_intensities, 25),
            'tumor_q75': np.percentile(tumor_intensities, 75),
            'tumor_iqr': np.percentile(tumor_intensities, 75) - np.percentile(tumor_intensities, 25),
        }
        
        return metrics
    
    def analyze_case(self, image_path, label_path):
        """Analyze a single case"""
        # Load data
        img_nib = nib.load(image_path)
        lbl_nib = nib.load(label_path)
        
        image = img_nib.get_fdata()
        mask = lbl_nib.get_fdata() > 0  # Binary mask
        
        spacing = img_nib.header.get_zooms()
        
        # Compute metrics
        metrics = self.compute_tumor_metrics(image, mask, spacing)
        if metrics is None:
            return None
        
        # Extract intensity profile
        profile = self.extract_intensity_profile(image, mask, spacing)
        
        # Add metadata
        metrics['case_name'] = image_path.stem
        metrics['profile'] = profile
        
        return metrics
    
    def analyze_dataset(self, n_workers=None):
        """Analyze entire dataset with optional parallelization
        
        Args:
            n_workers: Number of parallel workers. None=sequential, 0=all CPUs,
                      or specify an integer.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os
        
        matched_files = self.get_matched_files()
        print(f"Found {len(matched_files)} matched image-label pairs")
        
        self.results = []
        self.profiles = []
        
        if n_workers is None or n_workers == 1:
            # Sequential (original behavior)
            for img_path, lbl_path in tqdm(matched_files, desc="Analyzing cases"):
                result = self.analyze_case(img_path, lbl_path)
                if result is not None:
                    profile = result.pop('profile')
                    self.results.append(result)
                    self.profiles.append(profile)
        else:
            # Parallel
            if n_workers == 0:
                n_workers = os.cpu_count()
            print(f"Using {n_workers} parallel workers")
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_analyze_case_worker, img_path, lbl_path): i
                    for i, (img_path, lbl_path) in enumerate(matched_files)
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing cases"):
                    result = future.result()
                    if result is not None:
                        profile = result.pop('profile')
                        self.results.append(result)
                        self.profiles.append(profile)
        
        print(f"Successfully analyzed {len(self.results)} cases")
        return pd.DataFrame(self.results)
    
    def cluster_cases(self, n_clusters=3):
        """Cluster cases based on intensity characteristics and volume"""
        if len(self.results) == 0:
            print("No results to cluster. Run analyze_dataset() first.")
            return None
        
        # Select features for clustering
        # NOW INCLUDES VOLUME for augmentation pipeline determination
        feature_cols = [
            'tumor_mean', 'tumor_std', 'tumor_cv',
            'boundary_gradient_mean', 'tumor_to_background_contrast',
            'tumor_to_background_snr', 'tumor_volume_mm3'
        ]
        
        df = pd.DataFrame(self.results)
        X = df[feature_cols].values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df['cluster'] = clusters
        
        # Compute cluster statistics
        cluster_stats = df.groupby('cluster')[feature_cols].agg(['mean', 'std'])
        
        # Save model components for later use (e.g., applying to test set)
        self.clustering_model = kmeans
        self.scaler = scaler
        self.feature_columns = feature_cols
        
        return df, cluster_stats
    
    def select_optimal_clusters(self, k_range=range(2, 11), plot=True, save_path=None):
        """
        Automatically determine optimal number of clusters using multiple metrics
        
        Args:
            k_range: Range of k values to test (default: 2-10)
            plot: Whether to plot the validation metrics
            save_path: Path to save the plot (if plot=True)
        
        Returns:
            optimal_k: Recommended number of clusters
            metrics_df: DataFrame with all metrics for each k
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        print(f"  Testing k values from {min(k_range)} to {max(k_range)}...")
        
        # Prepare features - use same features as cluster_cases
        feature_cols = [
            'tumor_mean', 'tumor_std', 'tumor_cv',
            'boundary_gradient_mean', 'tumor_to_background_contrast',
            'tumor_to_background_snr', 'tumor_volume_mm3'
        ]
        
        df = pd.DataFrame(self.results)
        X = df[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store metrics
        metrics = {
            'k': [],
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        for k in k_range:
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Compute metrics
            metrics['k'].append(k)
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(X_scaled, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
            
            print(f"    k={k}: Silhouette={metrics['silhouette'][-1]:.3f}, "
                  f"CH={metrics['calinski_harabasz'][-1]:.1f}, "
                  f"DB={metrics['davies_bouldin'][-1]:.3f}")
        
        metrics_df = pd.DataFrame(metrics)
        
        # Determine optimal k using weighted scoring
        # Normalize metrics to [0, 1] range
        metrics_df['silhouette_norm'] = (metrics_df['silhouette'] - metrics_df['silhouette'].min()) / \
                                         (metrics_df['silhouette'].max() - metrics_df['silhouette'].min())
        
        metrics_df['ch_norm'] = (metrics_df['calinski_harabasz'] - metrics_df['calinski_harabasz'].min()) / \
                                 (metrics_df['calinski_harabasz'].max() - metrics_df['calinski_harabasz'].min())
        
        # Davies-Bouldin: lower is better, so invert
        metrics_df['db_norm'] = 1 - ((metrics_df['davies_bouldin'] - metrics_df['davies_bouldin'].min()) / \
                                      (metrics_df['davies_bouldin'].max() - metrics_df['davies_bouldin'].min()))
        
        # Elbow detection: compute second derivative of inertia
        inertia_diff = np.diff(metrics_df['inertia'].values)
        inertia_diff2 = np.diff(inertia_diff)
        elbow_k = list(k_range)[np.argmax(inertia_diff2) + 2] if len(inertia_diff2) > 0 else None
        
        # Combined score (weighted average)
        # Weights: Silhouette (40%), Calinski-Harabasz (30%), Davies-Bouldin (30%)
        metrics_df['combined_score'] = (
            0.40 * metrics_df['silhouette_norm'] +
            0.30 * metrics_df['ch_norm'] +
            0.30 * metrics_df['db_norm']
        )
        
        optimal_k = metrics_df.loc[metrics_df['combined_score'].idxmax(), 'k']
        
        print(f"\n  ✓ Optimal k based on combined metrics: {int(optimal_k)}")
        if elbow_k:
            print(f"  ✓ Elbow method suggests k: {elbow_k}")
        
        # Plot if requested — three individual figures for LaTeX \subfigure
        if plot:
            save_dir = Path(save_path).parent if save_path else Path('.')
            stem = Path(save_path).stem if save_path else 'cluster_selection'

            # Silhouette
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(metrics_df['k'], metrics_df['silhouette'], 'ko-', linewidth=2, markersize=8)
            ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.set_xticks(list(k_range))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f'{stem}_silhouette.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            # Calinski-Harabasz
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(metrics_df['k'], metrics_df['calinski_harabasz'], 'ko-', linewidth=2, markersize=8)
            ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Calinski\u2013Harabasz Index')
            ax.set_xticks(list(k_range))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f'{stem}_calinski.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            # Davies-Bouldin
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(metrics_df['k'], metrics_df['davies_bouldin'], 'ko-', linewidth=2, markersize=8)
            ax.axvline(optimal_k, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Davies\u2013Bouldin Index')
            ax.set_xticks(list(k_range))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / f'{stem}_davies_bouldin.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Saved 3 cluster selection plots to {save_dir}/")
        
        return int(optimal_k), metrics_df
    
    def plot_intensity_profiles(self, save_path='intensity_profiles.pdf', n_cases=None):
        """Plot intensity profiles — individual PDFs"""
        if len(self.profiles) == 0:
            print("No profiles to plot")
            return
        
        save_dir = Path(save_path).parent
        
        if n_cases is None or n_cases > len(self.profiles):
            n_cases = len(self.profiles)
        indices = np.linspace(0, len(self.profiles)-1, n_cases, dtype=int)
        
        df = pd.DataFrame(self.results)
        common_distances = np.linspace(-15, 15, 61)
        
        # 1. Individual profiles
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for idx in indices:
            profile = self.profiles[idx]
            ax.plot(profile['distance'], profile['mean'], alpha=0.6, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', label='Boundary', linewidth=2)
        ax.set_xlabel('Distance from boundary (mm)')
        ax.set_ylabel('Mean Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'intensity_profiles_individual.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Average profile with CI
        fig, ax = plt.subplots(figsize=(6, 4.5))
        interpolated_means = []
        for profile in self.profiles:
            interp_mean = np.interp(common_distances, profile['distance'], profile['mean'])
            interpolated_means.append(interp_mean)
        interpolated_means = np.array(interpolated_means)
        mean_profile = np.mean(interpolated_means, axis=0)
        std_profile = np.std(interpolated_means, axis=0)
        ax.plot(common_distances, mean_profile, 'b-', linewidth=2, label='Mean')
        ax.fill_between(common_distances, mean_profile - std_profile, mean_profile + std_profile,
                        alpha=0.3, label='\u00b11 SD')
        ax.axvline(x=0, color='red', linestyle='--', label='Boundary', linewidth=2)
        ax.set_xlabel('Distance from boundary (mm)')
        ax.set_ylabel('Mean Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'intensity_profiles_average.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Boundary gradient distribution
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.hist(df['boundary_gradient_mean'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Mean Boundary Gradient')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'boundary_gradient_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Contrast vs heterogeneity
        fig, ax = plt.subplots(figsize=(6, 4.5))
        scatter = ax.scatter(df['tumor_cv'], df['tumor_to_background_contrast'],
                           c=df['boundary_gradient_mean'], cmap='viridis',
                           s=100, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Tumor Heterogeneity (CV)')
        ax.set_ylabel('Tumor-to-Background Contrast')
        plt.colorbar(scatter, ax=ax, label='Boundary Gradient')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'contrast_vs_heterogeneity.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved 4 intensity profile plots to {save_dir}/")
    
    def plot_cluster_analysis(self, df, save_path='cluster_analysis.pdf'):
        """Plot cluster analysis — individual PDFs"""
        if 'cluster' not in df.columns:
            print("No cluster information. Run cluster_cases() first.")
            return
        
        save_dir = Path(save_path).parent
        common_distances = np.linspace(-15, 15, 61)
        
        # 1. Cluster distribution
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cluster_counts = df['cluster'].value_counts().sort_index()
        ax.bar(cluster_counts.index, cluster_counts.values, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Cases')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_dir / 'cluster_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Contrast vs heterogeneity by cluster
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            ax.scatter(cluster_data['tumor_cv'],
                      cluster_data['tumor_to_background_contrast'],
                      label=f'Cluster {cluster}', s=100, alpha=0.6, edgecolors='black')
        ax.set_xlabel('Tumor Heterogeneity (CV)')
        ax.set_ylabel('Tumor-to-Background Contrast')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'cluster_contrast_vs_heterogeneity.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Boundary gradient by cluster
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cluster_data_list = [df[df['cluster'] == c]['boundary_gradient_mean'].values
                            for c in sorted(df['cluster'].unique())]
        ax.boxplot(cluster_data_list, labels=[f'C{i}' for i in sorted(df['cluster'].unique())])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Boundary Gradient')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_dir / 'cluster_boundary_gradient.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Average intensity profiles by cluster
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for cluster in sorted(df['cluster'].unique()):
            cluster_indices = df[df['cluster'] == cluster].index
            cluster_profiles = [self.profiles[i] for i in cluster_indices if i < len(self.profiles)]
            if len(cluster_profiles) > 0:
                interpolated_means = []
                for profile in cluster_profiles:
                    interp_mean = np.interp(common_distances, profile['distance'], profile['mean'])
                    interpolated_means.append(interp_mean)
                mean_profile = np.mean(interpolated_means, axis=0)
                ax.plot(common_distances, mean_profile, linewidth=2, label=f'Cluster {cluster}')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Distance from boundary (mm)')
        ax.set_ylabel('Mean Intensity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'cluster_intensity_profiles.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved 4 cluster analysis plots to {save_dir}/")
    
    def generate_augmentation_insights(self, df):
        """Generate actionable insights for augmentation strategy"""
        print("\n" + "="*80)
        print("AUGMENTATION INSIGHTS")
        print("="*80)
        
        # Boundary sharpness
        mean_gradient = df['boundary_gradient_mean'].mean()
        std_gradient = df['boundary_gradient_mean'].std()
        
        print(f"\n1. BOUNDARY CHARACTERISTICS:")
        print(f"   - Mean boundary gradient: {mean_gradient:.2f} ± {std_gradient:.2f}")
        if mean_gradient > 50:  # Arbitrary threshold, adjust based on your data
            print("   ✓ Sharp boundaries detected")
            print("   → Recommendation: Can use aggressive spatial augmentations (elastic deform, rotation)")
        else:
            print("   ⚠ Soft/fuzzy boundaries detected")
            print("   → Recommendation: Use gentle spatial augmentations, avoid extreme elastic deformations")
        
        # Tumor heterogeneity
        mean_cv = df['tumor_cv'].mean()
        std_cv = df['tumor_cv'].std()
        
        print(f"\n2. TUMOR INTENSITY HETEROGENEITY:")
        print(f"   - Mean coefficient of variation: {mean_cv:.3f} ± {std_cv:.3f}")
        if mean_cv > 0.3:
            print("   ⚠ High heterogeneity detected")
            print("   → Recommendation: Preserve intensity variations, use moderate intensity augmentations")
        else:
            print("   ✓ Homogeneous tumors detected")
            print("   → Recommendation: Can use stronger intensity augmentations (gamma, brightness)")
        
        # Contrast
        mean_contrast = df['tumor_to_background_contrast'].mean()
        std_contrast = df['tumor_to_background_contrast'].std()
        
        print(f"\n3. TUMOR-TO-BACKGROUND CONTRAST:")
        print(f"   - Mean contrast ratio: {mean_contrast:.3f} ± {std_contrast:.3f}")
        if mean_contrast < 0.2:
            print("   ⚠ Low contrast detected")
            print("   → Recommendation: Consider contrast-enhancing augmentations (CLAHE, adaptive hist eq)")
        else:
            print("   ✓ Good contrast detected")
            print("   → Recommendation: Standard intensity augmentations sufficient")
        
        # Dataset variability
        if 'cluster' in df.columns:
            n_clusters = df['cluster'].nunique()
            cluster_balance = df['cluster'].value_counts(normalize=True)
            
            print(f"\n4. DATASET VARIABILITY:")
            print(f"   - Identified {n_clusters} distinct presentation clusters")
            print("   - Cluster distribution:")
            for cluster, proportion in cluster_balance.items():
                print(f"     Cluster {cluster}: {proportion*100:.1f}%")
            
            if cluster_balance.max() > 0.6:
                print("   ⚠ Imbalanced presentations detected")
                print("   → Recommendation: Oversample minority cluster cases or use targeted augmentations")
            else:
                print("   ✓ Balanced presentations detected")
                print("   → Recommendation: Standard augmentation pipeline appropriate")
        
        # Tumor size variability
        mean_volume = df['tumor_volume_mm3'].mean()
        std_volume = df['tumor_volume_mm3'].std()
        cv_volume = std_volume / mean_volume
        min_volume = df['tumor_volume_mm3'].min()
        max_volume = df['tumor_volume_mm3'].max()
        
        print(f"\n5. TUMOR SIZE VARIABILITY:")
        print(f"   - Mean volume: {mean_volume:.1f} ± {std_volume:.1f} mm³")
        print(f"   - Range: {min_volume:.1f} - {max_volume:.1f} mm³")
        print(f"   - CV: {cv_volume:.3f}")
        
        # Volume-based augmentation recommendations
        if min_volume < 1000:  # Small tumors present
            print("   ⚠ Small tumors detected (<1000 mm³)")
            print("   → Recommendation: Aggressive foreground oversampling (e.g., oversample_foreground_percent=0.33)")
            print("   → Use smaller patch sizes or ensure tumor-centered cropping")
            print("   → Gentle elastic deformations (scale=20, sigma=3) to avoid destroying small structures")
        
        if cv_volume > 0.5:
            print("   ⚠ High size variability detected")
            print("   → Recommendation: Multi-scale training approach")
            print("   → Use random cropping with various patch sizes (e.g., 128³, 160³, 192³)")
            print("   → Consider size-stratified sampling during training")
        else:
            print("   ✓ Consistent tumor sizes")
            print("   → Recommendation: Standard fixed patch size appropriate")
        
        # Cluster-specific volume analysis
        if 'cluster' in df.columns:
            print(f"\n   Cluster-specific volume characteristics:")
            for cluster in sorted(df['cluster'].unique()):
                cluster_vol = df[df['cluster'] == cluster]['tumor_volume_mm3']
                print(f"     Cluster {cluster}: {cluster_vol.mean():.0f} ± {cluster_vol.std():.0f} mm³ "
                      f"(range: {cluster_vol.min():.0f}-{cluster_vol.max():.0f})")
        
        print("\n" + "="*80)
        print("RECOMMENDED AUGMENTATION PIPELINE:")
        print("="*80)
        
        # Determine spatial augmentation intensity based on tumor size
        has_small_tumors = df['tumor_volume_mm3'].min() < 1000
        spatial_intensity = "gentle" if has_small_tumors else "moderate"
        
        print(f"""
Based on your dataset characteristics:

1. Spatial Augmentations ({spatial_intensity} due to {"small tumors present" if has_small_tumors else "tumor size distribution"}):
   - Rotation: ±15-30° (adjust based on boundary sharpness)
   - Elastic deformation: scale={'20, sigma=3 (gentle for small tumors)' if has_small_tumors else '30, sigma=5'}
   - Random scaling: 0.8-1.2
   - Random flipping: along appropriate anatomical axes
   - Mirror augmentation: Consider for anatomical symmetry

2. Intensity Augmentations:
   - Gamma correction: 0.7-1.5
   - Brightness: ±0.1 (adjust based on heterogeneity)
   - Contrast: 0.8-1.2
   - Gaussian noise: σ=0.01-0.05
   - Gaussian blur: σ=0.5-1.0 (simulate soft boundaries if needed)

3. Sampling Strategy (CRITICAL for small/variable tumors):
   - Foreground oversampling: {0.33 if has_small_tumors else 0.25} (fraction of samples guaranteed to contain tumor)
   - Patch size: {' 128³ or 160³ for small tumors' if has_small_tumors else '160³ or 192³'}
   {'   - Ensure tumor-centered cropping for cases with volume <1000 mm³' if has_small_tumors else ''}
   {'   - Multi-scale training with variable patch sizes (128³, 160³, 192³)' if cv_volume > 0.5 else ''}

4. Advanced Augmentations (if needed):
   - CLAHE (if low contrast detected)
   - SimCLR-style augmentations for self-supervised pretraining
   - MixUp/CutMix: Be cautious with small tumors

5. Class Balance & Loss:
   - Weighted loss functions (e.g., Dice + CE with class weights)
   {'   - Higher loss weight for small tumor voxels' if has_small_tumors else ''}
   - Consider region-based loss for boundary refinement

6. Cluster-Specific Strategies:""")
        
        if 'cluster' in df.columns:
            for cluster in sorted(df['cluster'].unique()):
                cluster_data = df[df['cluster'] == cluster]
                n_cases = len(cluster_data)
                vol = cluster_data['tumor_volume_mm3'].mean()
                contrast = cluster_data['tumor_to_background_contrast'].mean()
                cv = cluster_data['tumor_cv'].mean()
                
                print(f"   Cluster {cluster} (n={n_cases}, vol={vol:.0f}mm³, contrast={contrast:.2f}, CV={cv:.3f}):")
                
                if vol < 1000:
                    print(f"     → Small tumors: Aggressive foreground oversampling, gentler augmentations")
                if contrast < 0.2:
                    print(f"     → Low contrast: Use CLAHE, avoid extreme intensity augmentations")
                if cv > 0.3:
                    print(f"     → Heterogeneous: Preserve intensity variations, moderate augmentations")
                
                # Cluster balance recommendation
                cluster_fraction = n_cases / len(df)
                if cluster_fraction < 0.15:
                    print(f"     → Underrepresented ({cluster_fraction*100:.1f}%): Consider targeted oversampling")
        
        print()

    
    def save_summary_report(self, df, filename='dataset_summary.txt'):
        """Save summary statistics to file"""
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RECTAL CANCER MRI DATASET ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total cases analyzed: {len(df)}\n\n")
            
            f.write("INTENSITY STATISTICS:\n")
            f.write("-"*80 + "\n")
            stats_cols = [
                'tumor_mean', 'tumor_std', 'tumor_cv',
                'boundary_gradient_mean', 'tumor_to_background_contrast',
                'tumor_to_background_snr', 'tumor_volume_mm3'
            ]
            f.write(df[stats_cols].describe().to_string())
            f.write("\n\n")
            
            if 'cluster' in df.columns:
                f.write("CLUSTER ANALYSIS:\n")
                f.write("-"*80 + "\n")
                f.write(df.groupby('cluster')[stats_cols].mean().to_string())
                f.write("\n\n")
        
        print(f"Saved summary report to {filename}")


def _analyze_case_worker(image_path, label_path):
    """Top-level function for parallel execution (must be picklable)."""
    img_nib = nib.load(image_path)
    lbl_nib = nib.load(label_path)
    image = img_nib.get_fdata()
    mask = lbl_nib.get_fdata() > 0
    spacing = img_nib.header.get_zooms()

    analyzer = TumorIntensityAnalyzer.__new__(TumorIntensityAnalyzer)
    metrics = analyzer.compute_tumor_metrics(image, mask, spacing)
    if metrics is None:
        return None

    profile = analyzer.extract_intensity_profile(image, mask, spacing)
    metrics['case_name'] = Path(image_path).stem
    metrics['profile'] = profile
    return metrics


def main():
    """Main analysis pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze tumor intensity characteristics')
    parser.add_argument('--images', type=str, required=True, help='Path to images folder')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels folder')
    parser.add_argument('--clusters', type=int, default=3, help='Number of clusters')
    parser.add_argument('--output_dir', type=str, default='analysis/results/tumor', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TumorIntensityAnalyzer(args.images, args.labels)
    
    # Run analysis
    print("Starting dataset analysis...")
    df = analyzer.analyze_dataset()
    
    # Save raw results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / 'tumor_metrics.csv', index=False)
    print(f"Saved metrics to {output_dir / 'tumor_metrics.csv'}")
    
    # Plot intensity profiles
    analyzer.plot_intensity_profiles(save_path=output_dir / 'intensity_profiles.pdf')
    
    # Cluster analysis
    print(f"\nPerforming cluster analysis with {args.clusters} clusters...")
    df_clustered, cluster_stats = analyzer.cluster_cases(n_clusters=args.clusters)
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # Plot clusters
    analyzer.plot_cluster_analysis(df_clustered, save_path=output_dir / 'cluster_analysis.pdf')
    
    # Save clustered results
    df_clustered.to_csv(output_dir / 'tumor_metrics_clustered.csv', index=False)
    
    # Generate insights
    analyzer.generate_augmentation_insights(df_clustered)
    
    # Save summary
    analyzer.save_summary_report(df_clustered, filename=output_dir / 'dataset_summary.txt')
    
    print(f"\n{'='*80}")
    print("Analysis complete! Check output files:")
    print(f"  - {output_dir / 'tumor_metrics.csv'}")
    print(f"  - {output_dir / 'intensity_profiles.pdf'}")
    print(f"  - {output_dir / 'cluster_analysis.pdf'}")
    print(f"  - {output_dir / 'dataset_summary.txt'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
