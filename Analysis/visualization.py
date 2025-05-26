import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    """
    A class for visualizing and analyzing student performance data.
    """
    
    def __init__(self):
        """Initialize the DataVisualizer class"""
        plt.style.use('seaborn-v0_8-deep')  # Set default style
        plt.grid(True)  # Enable grid
        
    def _prepare_data(self, df):
        """Helper method to validate and prepare data"""
        if 'status' not in df.columns:
            raise ValueError("DataFrame must contain a 'status' column")
        return [col for col in df.columns if col != 'status']

    def create_status_boxplot(self, df, width=15, height=6, title='Box Plot Features by Status'):
        """Create a seaborn boxplot showing feature distributions categorized by status."""
        plt.figure(figsize=(width, height))
        sns.boxplot(data=df.melt(id_vars=['status'], var_name='Module', value_name='Value'), 
                   x='Module', y='Value', hue='status',
                   flierprops={'markerfacecolor': '#FFA500', 'marker': 'o'},
                   palette=['cyan', 'orange'],
                   width=0.7,
                   linewidth=1.5)
        self._customize_plot(title)

    def create_boxplot(self, df, width=15, height=6, title='Box Plot of Features'):
        """Create a simple boxplot for all numeric columns."""
        module_columns = self._prepare_data(df)
        plt.figure(figsize=(width, height))
        df[module_columns].boxplot(patch_artist=True)
        self._customize_plot(title)

    def plot_density_skewness(self, df, width=15, height=6, n_cols=3):
        """Create density plots showing distribution skewness."""
        module_columns = self._prepare_data(df)
        n_features = len(module_columns)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height * (n_rows/2)))
        fig.suptitle('Density Plots Showing Distribution Skewness', y=1.02, fontsize=12)
        axes = axes.flatten()

        for idx, feature in enumerate(module_columns):
            skew = df[feature].skew()
            sns.kdeplot(data=df[feature], ax=axes[idx], fill=True)
            
            mean = df[feature].mean()
            median = df[feature].median()
            axes[idx].axvline(mean, color='red', linestyle='--', label='Mean')
            axes[idx].axvline(median, color='green', linestyle='--', label='Median')
            
            if abs(skew) <= 0.5:
                color = 'green'
            elif abs(skew) <= 1:
                color = 'orange'
            else:
                color = 'red'
                
            axes[idx].set_title(f'{feature}\nSkewness: {skew:.3f}', color=color)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        for idx in range(len(module_columns), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    def plot_passed_students_pie(self, df, width=15, height=6):
        """Create pie charts for passed students distribution."""
        self._plot_student_pie(df, passed=True, width=width, height=height)

    def plot_failed_students_pie(self, df, width=15, height=6):
        """Create pie charts for failed students distribution."""
        self._plot_student_pie(df, passed=False, width=width, height=height)

    def plot_bar_matplotlib(self, df, width=15, height=6):
        """
        Create stacked bar plot for student performance.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing module scores and status
        width : int, default=15
            Width of the plot
        height : int, default=6
            Height of the plot
        """
        module_columns = self._prepare_data(df)
        data = {
            module: {
                'Fail/Unenrolled and Dropout': sum((df[module] >= 5) & (df['status'] == 1)),
                'Passed': sum(df[module] < 5),
                'Fail': sum(df[module] == 5),
            }
            for module in module_columns
        }
        
        df_plot = pd.DataFrame(data).T
        fig, ax = plt.subplots(figsize=(width, height))
        
        bottom = np.zeros(len(df_plot))
        colors = ['steelblue', 'cyan', 'orange']
        
        # Create bars and store them for legend
        bars = []
        for i, col in enumerate(df_plot.columns):
            bar = ax.bar(df_plot.index, df_plot[col], bottom=bottom, label=col, color=colors[i])
            bars.append(bar)
            # Add value labels
            for j, v in enumerate(df_plot[col]):
                if v > 0:
                    ax.text(j, bottom[j] + v/2, str(int(v)), ha='center', va='center')
            bottom += df_plot[col]
        
        # Add legend with custom colors and labels
        plt.legend(
            title='Student Status',
            bbox_to_anchor=(1.05, 1),  # Position legend outside plot
            loc='upper left',
            borderaxespad=0.
        )
        
        # Customize plot with adjusted layout for legend
        plt.title("Student Performance in Each Module", fontsize=12, pad=15)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylabel('Number of Students')
        
        # Adjust layout to prevent label cutoff and accommodate legend
        plt.tight_layout()
        plt.show()

    def get_status_statistics(self, df):
        """Calculate descriptive statistics grouped by status."""
        module_columns = self._prepare_data(df)
        melted_df = df.melt(id_vars=['status'], 
                           value_vars=module_columns,
                           var_name='feature', 
                           value_name='value')
        
        stats = melted_df.groupby(['feature', 'status'])['value'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('min', 'min'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75)),
            ('max', 'max'),
            ('iqr', lambda x: x.quantile(0.75) - x.quantile(0.25))
        ]).round(3)
        
        return stats

    def calculate_dispersion_measures(self, df):
        """Calculate measures of dispersion for each feature."""
        module_columns = self._prepare_data(df)
        results = []
        
        for column in module_columns:
            results.append({
                'Feature': column,
                'Range': round(df[column].max() - df[column].min(), 3),
                'Variance': round(df[column].var(), 3),
                'Std_Dev': round(df[column].std(), 3),
                'IQR': round(df[column].quantile(0.75) - df[column].quantile(0.25), 3)
            })
        
        return pd.DataFrame(results)

    def calculate_distribution_shape(self, df):
        """Calculate distribution shape statistics."""
        module_columns = self._prepare_data(df)
        results = []
        
        for column in module_columns:
            skew = df[column].skew()
            kurt = df[column].kurtosis()
            
            if skew < -1:
                skew_interpret = "Highly Negative (Strong Left Tail)"
            elif -1 <= skew < -0.5:
                skew_interpret = "Moderately Negative (Left Tail)"
            elif -0.5 <= skew <= 0.5:
                skew_interpret = "Approximately Symmetric"
            elif 0.5 < skew <= 1:
                skew_interpret = "Moderately Positive (Right Tail)"
            else:
                skew_interpret = "Highly Positive (Strong Right Tail)"
                
            if kurt < -1:
                kurt_interpret = "Very Platykurtic (Very Light-tailed)"
            elif -1 <= kurt < 0:
                kurt_interpret = "Platykurtic (Light-tailed)"
            elif -0 <= kurt < 1:
                kurt_interpret = "Mesokurtic (Normal-tailed)"
            else:
                kurt_interpret = "Leptokurtic (Heavy-tailed)"
                
            results.append({
                'Feature': column,
                'Skewness': round(skew, 3),
                'Skewness_Interpretation': skew_interpret,
                'Kurtosis': round(kurt, 3),
                'Kurtosis_Interpretation': kurt_interpret
            })
            
        return pd.DataFrame(results)
    
    def calculate_central_tendency(self, df):
        """Calculate measures of central tendency (mean, median, mode)."""
        module_columns = self._prepare_data(df)
        results = []
        
        for column in module_columns:
            mode_value = df[column].mode()
            mode_str = f"{mode_value[0]:.3f}" if len(mode_value) == 1 else "Multiple"
            
            results.append({
                'Feature': column,
                'Mean': round(df[column].mean(), 3),
                'Median': round(df[column].median(), 3),
                'Mode': mode_str
            })
            
        return pd.DataFrame(results)

    def _plot_student_pie(self, df, passed=True, width=15, height=6):
        """Helper method for creating pie charts"""
        module_columns = self._prepare_data(df)
        n_modules = len(module_columns)
        n_cols = 4
        n_rows = (n_modules + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height * (n_rows/2)))
        title = "Distribution of Passed Students by Module" if passed else "Distribution of Failed Students by Module"
        fig.suptitle(title, y=1.02, fontsize=14)
        
        axes = axes.flatten()
        colors = ['skyblue', 'steelblue']
        
        for idx, module in enumerate(module_columns):
            if passed:
                val_graduate = sum((df[module] < 5) & (df['status'] == 0))
                val_dropout = sum((df[module] < 5) & (df['status'] == 1))
            else:
                val_graduate = sum((df[module] >= 5) & (df['status'] == 0))
                val_dropout = sum((df[module] >= 5) & (df['status'] == 1))
                
            values = [val_graduate, val_dropout]
            
            if sum(values) > 0:
                axes[idx].pie(values, 
                             labels=['Graduate', 'Dropout'],
                             colors=colors,
                             autopct='%1.1f%%',
                             startangle=90)
                
            axes[idx].set_title(f'{module}')
        
        for idx in range(len(module_columns), len(axes)):
            fig.delaxes(axes[idx])
            
        status_type = "Passed" if passed else "Failed"
        fig.legend([f'{status_type} (Graduate)', f'{status_type} (Dropout)'],
                   loc='center right',
                   bbox_to_anchor=(0.98, 0.5))
        
        plt.tight_layout()
        plt.show()

    def _customize_plot(self, title):
        """Helper method for common plot customization"""
        plt.title(title, fontsize=12, pad=15)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Example usage:
# visualizer = DataVisualizer()
# visualizer.create_status_boxplot(BHT_1)
# visualizer.plot_passed_students_pie(BHT_1)
# stats = visualizer.get_status_statistics(BHT_1)