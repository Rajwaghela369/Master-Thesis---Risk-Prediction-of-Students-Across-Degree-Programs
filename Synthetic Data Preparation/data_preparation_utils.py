import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import StratifiedShuffleSplit

def compare_distributions(orginal_df, synthetic_test, figsize=(15, 10)):
    """
    Create density plots comparing distributions between original and synthetic data for all columns.
    
    Parameters:
    -----------
    orginal_df : pandas.DataFrame
        Original test data
    synthetic_test : pandas.DataFrame
        Synthetic test data
    figsize : tuple, default=(15, 10)
        Figure size as (width, height)
    """
    # Get number of columns
    n_cols = len(orginal_df.columns)
    
    # Calculate number of rows needed (2 columns per row)
    n_rows = (n_cols + 1) // 2
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    fig.suptitle('Distribution Comparison: Original vs Synthetic Data', fontsize=14, y=1.02)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Create density plots for each feature
    for idx, column in enumerate(orginal_df.columns):
        # Plot densities
        sns.kdeplot(data=orginal_df[column], ax=axes[idx], label='Original', fill=True, alpha=0.5)
        sns.kdeplot(data=synthetic_test[column], ax=axes[idx], label='Synthetic', fill=True, alpha=0.5)
        
        # Add mean lines
        orig_mean = orginal_df[column].mean()
        syn_mean = synthetic_test[column].mean()
        axes[idx].axvline(orig_mean, color='blue', linestyle='--', label=f'Original Mean: {orig_mean:.2f}')
        axes[idx].axvline(syn_mean, color='orange', linestyle='--', label=f'Synthetic Mean: {syn_mean:.2f}')
        
        # Customize subplot
        axes[idx].set_title(f'{column}')
        axes[idx].set_xlabel('Values')
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Remove empty subplots if any
    for idx in range(len(orginal_df.columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


def generate_synthetic_data(original_data, portion=0.3):
    """
    Generate synthetic data using CTGANSynthesizer with specified constraints.

    Parameters:
    -----------
    original_data : pandas.DataFrame
        Original dataset to base synthetic data on
    portion : float, default=0.3
        Proportion of original data size to generate as synthetic data

    Returns:
    --------
    tuple
        (synthetic_data, metadata) - Synthetic dataset and its metadata
    """
    # Create metadata
    metadata = SingleTableMetadata()
    
    # Detect the types of columns automatically
    metadata.detect_from_dataframe(original_data)
    
    # Set 'status' column as categorical
    metadata.update_column('status', sdtype='categorical')

    # Create and configure synthesizer
    synthesizer = CTGANSynthesizer(metadata, epochs=100)
    
    # Fit the synthesizer with the original data
    synthesizer.fit(original_data)
    
    # Calculate number of synthetic samples
    n_synthetic = int(len(original_data) * portion)
    
    # Generate synthetic data
    synthetic_data = synthesizer.sample(num_rows=n_synthetic)
    
    # Print distribution comparison
    print("Original data distribution:", dict(original_data['status'].value_counts()))
    print("Synthetic data distribution:", dict(synthetic_data['status'].value_counts()))
    print("\nOriginal data shape:", original_data.shape)
    print("Synthetic data shape:", synthetic_data.shape)
    
    return synthetic_data, metadata



def merge_real_synthetic(real_data, synthetic_data):
    """
    Merge real data with synthetic data into one dataframe.
    
    Parameters:
    -----------
    real_data : pandas.DataFrame
        Original dataframe
    synthetic_data : pandas.DataFrame
        Synthetic dataframe 
    
    Returns:
    --------
    pandas.DataFrame
        Merged dataframe containing both real and synthetic data
    """
    # Merge real and synthetic data
    merged_df = pd.concat([real_data, synthetic_data], axis=0, ignore_index=True)

    # Check distribution of status column
    status_dist = merged_df['status'].value_counts()
    print("Distribution of status column in merged data:")
    print(status_dist)

    return merged_df

def save_merged_data(merged_df, dataset_name, base_dir=None):
    """
    Save merged dataframe to a CSV file.
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        Merged dataframe to save
    dataset_name : str
        Name of the dataset (used in filename)
    base_dir : str or Path, optional
        Base directory for saving file. If None, uses default path.
    """
    if base_dir is None:
        base_dir = Path('/Users/rajwaghela/Library/CloudStorage/GoogleDrive-rajwaghela4244@gmail.com/My Drive/Thesis/Practical/Data/Processed Data')
    else:
        base_dir = Path(base_dir)
    
    # Generate filename
    merged_file = base_dir / 'Merged' / f'{dataset_name}_merged.csv'
    
    # Create directory if it doesn't exist
    merged_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save file
        merged_df.to_csv(merged_file, index=False)
        
    except Exception as e:
        print(f"Error saving file: {str(e)}")

# Optional: Add a main block for testing
if __name__ == "__main__":
    # Add any testing code here
    pass