# utils/visual_utils.py
# Author: Saeed Ur Rehman
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Union, List, Any
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator # For controlling number of ticks
import logging

logger = logging.getLogger(__name__)
# Apply a base theme. Specific plot functions can override palette.
sns.set_theme(style="whitegrid", palette="muted") 

class VisualUtils:
    @staticmethod
    def _create_base_figure(title: str, figsize: tuple = (9, 6)) -> tuple[Figure, Any]: # Any for plt.Axes
        """
        Creates a base Matplotlib figure and axes with a title and basic styling.
        Uses Matplotlib's Figure directly for better integration with PyQt.
        """
        fig = Figure(figsize=figsize, dpi=90) # Slightly lower DPI for performance if needed
        fig.patch.set_facecolor('#f8f9fa') 
        ax = fig.add_subplot(111)
        ax.set_facecolor('#ffffff') 
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10, color="#333333") # Darker title
        ax.tick_params(axis='both', which='major', labelsize=9, colors="#555555")
        for spine_pos in ['top', 'right']:
            ax.spines[spine_pos].set_visible(False)
        for spine_pos in ['left', 'bottom']:
            ax.spines[spine_pos].set_color('#cccccc') # Lighter spine color
        return fig, ax

    @staticmethod
    def _validate_columns(data: pd.DataFrame, 
                          required_cols: Optional[List[str]] = None, 
                          numeric_cols: Optional[List[str]] = None, 
                          categorical_cols: Optional[List[str]] = None):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        
        active_required_cols = [col for col in (required_cols or []) if col] # Filter out None

        if data.empty and active_required_cols:
            raise ValueError("Input DataFrame is empty, cannot plot specified columns.")

        for col in active_required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame.")

        if numeric_cols:
            for col in filter(None, numeric_cols):
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column '{col}' must be numeric for this plot type.")
        if categorical_cols:
            for col in filter(None, categorical_cols):
                 if col in data.columns and not pd.api.types.is_categorical_dtype(data[col]) and \
                                           not pd.api.types.is_object_dtype(data[col]):
                    raise ValueError(f"Column '{col}' must be categorical or object type.")

    @staticmethod
    def create_scatter_plot(data: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str] = None, title: Optional[str] = None) -> Figure:
        VisualUtils._validate_columns(data, [x_col, y_col], numeric_cols=[x_col, y_col])
        if hue: VisualUtils._validate_columns(data, [hue])

        plot_title = title if title else f"Scatter: {y_col} vs. {x_col}" + (f" (by {hue})" if hue else "")
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(8.5, 5.5))
        
        # Reduce point size and alpha for potentially dense plots
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue, ax=ax, alpha=0.6, edgecolor=None, s=30, palette="viridis" if hue else None)
        ax.set_xlabel(x_col, fontsize=10); ax.set_ylabel(y_col, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5) # Lighter grid
        if hue and data[hue].nunique() < 12: # Legend for reasonable number of categories
            ax.legend(title=hue, frameon=True, fontsize=8, facecolor='#ffffffE0', edgecolor='#cccccc') # Legend with background
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_box_plot(data: pd.DataFrame, x_col_categorical: str, y_col_numeric: str, hue: Optional[str] = None, title: Optional[str] = None) -> Figure:
        VisualUtils._validate_columns(data, [x_col_categorical, y_col_numeric], numeric_cols=[y_col_numeric], categorical_cols=[x_col_categorical])
        if hue: VisualUtils._validate_columns(data, [hue], categorical_cols=[hue])

        plot_title = title if title else f"Box Plot: {y_col_numeric} by {x_col_categorical}" + (f" (Grouped by {hue})" if hue else "")
        
        num_x_cats = data[x_col_categorical].nunique()
        num_hue_cats = data[hue].nunique() if hue else 1
        # Adjust figure width based on number of categories and hue groups
        fig_width = max(7, min(20, num_x_cats * (1.0 + num_hue_cats * 0.3) ) )
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(fig_width, 5.5))
        
        sns.boxplot(data=data, x=x_col_categorical, y=y_col_numeric, hue=hue, ax=ax, palette="Set2", linewidth=1.0, fliersize=3) # Smaller fliers
        ax.set_xlabel(x_col_categorical, fontsize=10); ax.set_ylabel(y_col_numeric, fontsize=10)
        
        if num_x_cats > 6: ax.tick_params(axis='x', rotation=30, ha='right', labelsize=8.5)
        elif num_x_cats > 10: ax.tick_params(axis='x', rotation=45, ha='right', labelsize=8)

        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        if hue and num_hue_cats < 10: ax.legend(title=hue, frameon=True, fontsize=8, facecolor='#ffffffE0', edgecolor='#cccccc')
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_correlation_matrix(data: pd.DataFrame, title: str = "Correlation Matrix") -> Figure:
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty or numeric_data.shape[1] < 2:
            fig, ax = VisualUtils._create_base_figure("Correlation Matrix: Insufficient Data", figsize=(7,5))
            ax.text(0.5, 0.5, "At least two numeric columns required.", ha='center', va='center', color='grey')
            return fig
        
        num_cols = numeric_data.shape[1]
        fig_size = max(6, min(15, num_cols * 0.6)) # Cap max size
        show_annot = num_cols <= 12 # Annotate only for smaller matrices for readability
        annot_kws_size = max(6, 9 - num_cols // 2.5)
        tick_fontsize = max(7, 9.5 - num_cols // 3)

        fig, ax = VisualUtils._create_base_figure(title, figsize=(fig_size, fig_size * 0.8))
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=show_annot, cmap="coolwarm", fmt=".2f" if show_annot else None, 
                    ax=ax, linewidths=.3, cbar_kws={"shrink": .75}, annot_kws={"size": annot_kws_size})
        
        ax.tick_params(axis='x', rotation=45, ha='right', labelsize=tick_fontsize)
        ax.tick_params(axis='y', rotation=0, labelsize=tick_fontsize)
        try: fig.tight_layout(pad=1.2) # More padding for heatmap labels
        except ValueError: logger.warning("Tight layout failed for correlation matrix.")
        return fig

    @staticmethod
    def create_pairplot(data: pd.DataFrame, hue: Optional[str] = None, max_numeric_cols: int = 5, title: str = "Pair Plot") -> Figure:
        numeric_cols_df = data.select_dtypes(include=np.number)
        if numeric_cols_df.shape[1] < 2:
            fig, ax = VisualUtils._create_base_figure("Pair Plot: Insufficient Numeric Data", figsize=(7,5))
            ax.text(0.5, 0.5, "At least two numeric columns required.", ha='center', va='center', color='grey')
            return fig

        cols_for_plot = numeric_cols_df.columns.tolist()
        if len(cols_for_plot) > max_numeric_cols:
            logger.info(f"Pair plot: Using first {max_numeric_cols} of {len(cols_for_plot)} numeric columns.")
            cols_for_plot = cols_for_plot[:max_numeric_cols]
        
        plot_data = data[cols_for_plot].copy() # Data with selected numeric columns
        actual_hue_col = None
        if hue and hue in data.columns:
            plot_data[hue] = data[hue] # Add hue column
            actual_hue_col = hue
            if not (pd.api.types.is_categorical_dtype(data[hue]) or pd.api.types.is_object_dtype(data[hue])):
                 logger.warning(f"Hue column '{hue}' for pair plot is numeric. Consider using a categorical hue.")

        g = sns.pairplot(plot_data, hue=actual_hue_col, diag_kind='hist', corner=False, # Changed diag_kind, corner
                         plot_kws={'alpha':0.5, 's':20, 'edgecolor':None}, # Smaller points, no edge
                         diag_kws={'alpha':0.6, 'bins': 20}) # Fewer bins for diag hist
        g.fig.suptitle(title + (f" (by {actual_hue_col})" if actual_hue_col else ""), y=1.015, fontsize=14) # Adjust y for title
        g.fig.patch.set_facecolor('#f8f9fa')
        for r_idx, row_axes in enumerate(g.axes):
            for c_idx, ax_iter in enumerate(row_axes):
                if ax_iter is not None:
                    ax_iter.set_facecolor('#ffffff')
                    ax_iter.tick_params(colors="#555555", labelsize=7) # Smaller ticks for pairplot
                    ax_iter.xaxis.label.set_color("#333333"); ax_iter.xaxis.label.set_size(9)
                    ax_iter.yaxis.label.set_color("#333333"); ax_iter.yaxis.label.set_size(9)
                    if r_idx != c_idx : ax_iter.grid(True, linestyle=':', alpha=0.4) # Grid only on scatter
        if actual_hue_col and data[actual_hue_col].nunique() > 10: g.legend.set_visible(False) # Hide legend if too many hue items
        
        try: plt.gcf().subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.96) # Manual adjustment
        except: pass # Fails if no current figure, g.fig should be current
        return g.fig

    @staticmethod
    def create_distribution_plot(data: pd.DataFrame, col_name: str, title: Optional[str] = None) -> Figure: # Same as create_histogram with KDE
        return VisualUtils.create_histogram(data[col_name], bins='auto', kde=True, title=title or f"Distribution of {col_name}")

    @staticmethod
    def create_kde_plot(series: pd.Series, title: Optional[str] = None) -> Figure:
        if not pd.api.types.is_numeric_dtype(series): raise ValueError(f"KDE requires numeric data. '{series.name}' is not.")
        if series.isnull().all() or len(series.dropna()) < 2 :
            fig, ax = VisualUtils._create_base_figure(f"KDE: Insufficient Data for {series.name}", figsize=(8,5))
            ax.text(0.5, 0.5, "Not enough data points for KDE.", ha='center', va='center', color='grey')
            return fig

        plot_title = title if title else f"Kernel Density Estimate for {series.name}"
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(8.5,5))
        sns.kdeplot(series.dropna(), fill=True, ax=ax, linewidth=1.5, color=sns.color_palette("crest_r", 1)[0], alpha=0.6)
        ax.set_xlabel(series.name, fontsize=10); ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_histogram(series: pd.Series, bins: Union[int, str] = 'fd', kde: bool = False, title: Optional[str] = None) -> Figure: # fd rule for bins
        if not pd.api.types.is_numeric_dtype(series): raise ValueError(f"Histogram requires numeric data. '{series.name}' is not.")
        if series.isnull().all():
            fig, ax = VisualUtils._create_base_figure(f"Histogram: No Data for {series.name}", figsize=(8,5))
            ax.text(0.5, 0.5, "No data to plot histogram.", ha='center', va='center', color='grey')
            return fig

        plot_title = title if title else f"Histogram of {series.name}" + (" with KDE" if kde else "")
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(8.5,5))
        sns.histplot(series.dropna(), kde=kde, ax=ax, bins=bins, color=sns.color_palette("flare",1)[0], edgecolor=None, alpha=0.7, stat="density" if kde else "count")
        ax.set_xlabel(series.name, fontsize=10); ax.set_ylabel("Density" if kde else "Frequency", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        if kde: ax.lines[0].set_linewidth(1.5) # Style KDE line if present
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_violin_plot(data: pd.DataFrame, x_col_categorical: str, y_col_numeric: str, hue: Optional[str] = None, title: Optional[str] = None) -> Figure:
        VisualUtils._validate_columns(data, [x_col_categorical, y_col_numeric], numeric_cols=[y_col_numeric], categorical_cols=[x_col_categorical])
        if hue: VisualUtils._validate_columns(data, [hue], categorical_cols=[hue])

        plot_title = title if title else f"Violin Plot: {y_col_numeric} by {x_col_categorical}" + (f" (Grouped by {hue})" if hue else "")
        num_x_cats = data[x_col_categorical].nunique()
        fig_width = max(7, min(18, num_x_cats * (0.8 + (data[hue].nunique() if hue else 1) * 0.25) ) )
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(fig_width, 5.5))
        
        sns.violinplot(data=data, x=x_col_categorical, y=y_col_numeric, hue=hue, ax=ax, palette="Set3", inner="quart", linewidth=1.0, cut=0, density_norm="width", bw_method=0.2) # density_norm, bw_method
        ax.set_xlabel(x_col_categorical, fontsize=10); ax.set_ylabel(y_col_numeric, fontsize=10)
        if num_x_cats > 6: ax.tick_params(axis='x', rotation=30, ha='right', labelsize=8.5)
        elif num_x_cats > 10: ax.tick_params(axis='x', rotation=45, ha='right', labelsize=8)
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        if hue and data[hue].nunique() < 10: ax.legend(title=hue, frameon=True, fontsize=8, facecolor='#ffffffE0', edgecolor='#cccccc')
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_line_plot(data: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str] = None, title: Optional[str] = None, parse_x_as_date: bool = True) -> Figure:
        VisualUtils._validate_columns(data, [x_col, y_col], numeric_cols=[y_col]) # X can be date/time or numeric
        if hue: VisualUtils._validate_columns(data, [hue])
        
        plot_data = data.copy(); is_datetime_x = False
        if parse_x_as_date and x_col in plot_data.columns:
            try: plot_data[x_col] = pd.to_datetime(plot_data[x_col]); is_datetime_x = True
            except: logger.warning(f"Could not parse '{x_col}' as datetime for line plot.")
        if not pd.api.types.is_numeric_dtype(plot_data[x_col]) and not is_datetime_x:
            raise ValueError(f"X-axis '{x_col}' for line plot must be numeric or datetime.")

        plot_data.sort_values(by=x_col, inplace=True)
        plot_title = title if title else f"Line: {y_col} over {x_col}" + (f" (by {hue})" if hue else "")
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(9.5,5.5))
        
        sns.lineplot(data=plot_data, x=x_col, y=y_col, hue=hue, ax=ax, marker='o', linewidth=1.2, markersize=4, palette="tab10" if hue else None, errorbar=None) # Smaller marker, no errorbar for cleaner line
        ax.set_xlabel(x_col, fontsize=10); ax.set_ylabel(y_col, fontsize=10)
        if is_datetime_x: fig.autofmt_xdate(rotation=25, ha='right')
        elif plot_data[x_col].nunique() > 12 : ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both')) # Limit numeric ticks
        ax.grid(True, linestyle=':', alpha=0.6)
        if hue and plot_data[hue].nunique() < 12: ax.legend(title=hue, frameon=True, fontsize=8, facecolor='#ffffffE0', edgecolor='#cccccc')
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_area_plot(data: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str] = None, title: Optional[str] = None, parse_x_as_date: bool = True) -> Figure:
        VisualUtils._validate_columns(data, [x_col, y_col], numeric_cols=[y_col])
        if hue: VisualUtils._validate_columns(data, [hue])
        plot_data = data.copy(); is_datetime_x = False
        if parse_x_as_date and x_col in plot_data.columns:
            try: plot_data[x_col] = pd.to_datetime(plot_data[x_col]); is_datetime_x = True
            except: logger.warning(f"Could not parse '{x_col}' as datetime for area plot.")
        plot_data.sort_values(by=x_col, inplace=True)
        plot_title = title if title else f"Area Plot: {y_col} over {x_col}" + (f" (by {hue})" if hue else "")
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(9.5,5.5))

        if hue and hue in plot_data.columns and plot_data[hue].nunique() > 1:
            unique_hues = plot_data[hue].unique()[:10] # Limit number of hues plotted directly for clarity
            if len(unique_hues) < plot_data[hue].nunique():
                 logger.warning(f"Area plot: Showing first {len(unique_hues)} of {plot_data[hue].nunique()} hue categories.")
            palette = sns.color_palette("muted", n_colors=len(unique_hues))
            for i, h_val in enumerate(unique_hues):
                hue_data = plot_data[plot_data[hue] == h_val]
                ax.plot(hue_data[x_col], hue_data[y_col], label=str(h_val), color=palette[i], linewidth=1.2)
                ax.fill_between(hue_data[x_col], hue_data[y_col], alpha=0.15, color=palette[i]) # Lighter fill
        else:
            ax.fill_between(plot_data[x_col], plot_data[y_col], alpha=0.4, color=sns.color_palette("pastel")[3])
            ax.plot(plot_data[x_col], plot_data[y_col], color=sns.color_palette("pastel")[3], alpha=0.7, linewidth=1.2)
        
        ax.set_xlabel(x_col, fontsize=10); ax.set_ylabel(y_col, fontsize=10)
        if is_datetime_x: fig.autofmt_xdate(rotation=25, ha='right')
        elif plot_data[x_col].nunique() > 12 : ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))
        ax.grid(True, linestyle=':', alpha=0.6)
        if hue and plot_data[hue].nunique() < 12: ax.legend(title=hue, frameon=True, fontsize=8, facecolor='#ffffffE0', edgecolor='#cccccc')
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_bar_plot(data: pd.DataFrame, x_col_categorical: str, y_col_numeric: str, hue: Optional[str] = None, estimator=np.mean, title: Optional[str] = None) -> Figure:
        VisualUtils._validate_columns(data, [x_col_categorical, y_col_numeric], numeric_cols=[y_col_numeric], categorical_cols=[x_col_categorical])
        if hue: VisualUtils._validate_columns(data, [hue], categorical_cols=[hue])

        agg_func_name = estimator.__name__ if hasattr(estimator, '__name__') else 'Value'
        plot_title = title if title else f"Bar: {agg_func_name} of {y_col_numeric} by {x_col_categorical}" + (f" (by {hue})" if hue else "")
        
        num_x_cats = data[x_col_categorical].nunique()
        num_hue_cats = data[hue].nunique() if hue else 1
        fig_width = max(7, min(20, num_x_cats * (0.6 + num_hue_cats * 0.25) ) ) # Dynamic width
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(fig_width, 5.5))
        
        try:
            sns.barplot(data=data, x=x_col_categorical, y=y_col_numeric, hue=hue, ax=ax, palette="viridis", estimator=estimator, errorbar=None)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error creating bar plot:\n{str(e)[:100]}", transform=ax.transAxes, color='red', ha='center', va='center', wrap=True)
            logger.error(f"Bar plot generation error: {e}", exc_info=True); return fig

        ax.set_xlabel(x_col_categorical, fontsize=10)
        ax.set_ylabel(f"{agg_func_name} of {y_col_numeric}", fontsize=10)
        if num_x_cats > 6: ax.tick_params(axis='x', rotation=30, ha='right', labelsize=8.5)
        elif num_x_cats > 10: ax.tick_params(axis='x', rotation=45, ha='right', labelsize=8)
        
        ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        if hue and num_hue_cats < 10: ax.legend(title=hue, frameon=True, fontsize=8, facecolor='#ffffffE0', edgecolor='#cccccc')
        fig.tight_layout(pad=0.7)
        return fig

    @staticmethod
    def create_pie_chart(data: pd.DataFrame, category_col: str, value_col: Optional[str] = None, max_slices: int = 7, title: Optional[str] = None) -> Figure: # Reduced max_slices for clarity
        VisualUtils._validate_columns(data, [category_col])
        if value_col: VisualUtils._validate_columns(data, [value_col], numeric_cols=[value_col])

        plot_title = title if title else f"Pie Chart: {category_col}" + (f" by {value_col}" if value_col else " by Count")
        fig, ax = VisualUtils._create_base_figure(plot_title, figsize=(7,7)) # Square figure

        if value_col:
            if not pd.api.types.is_numeric_dtype(data[value_col]):
                raise ValueError(f"Value column '{value_col}' must be numeric for pie sums.")
            series_data = data[data[value_col] > 0].groupby(category_col)[value_col].sum() # Sum positive values
        else:
            series_data = data[category_col].value_counts()

        if series_data.empty or (series_data <= 1e-9).all(): # Check for effectively zero sum
             ax.text(0.5, 0.5, "No positive data to plot pie chart.", transform=ax.transAxes, color='grey', ha='center'); return fig

        if len(series_data) > max_slices:
            series_data = series_data.sort_values(ascending=False)
            top_n = series_data.head(max_slices - 1)
            other_sum = series_data.iloc[max_slices - 1:].sum()
            if other_sum > 1e-9: 
                top_n[f"Other ({len(series_data)-(max_slices-1)})"] = other_sum
            series_data = top_n
        
        if series_data.empty: 
             ax.text(0.5, 0.5, "No data after filtering for pie chart.", transform=ax.transAxes, color='grey', ha='center'); return fig

        explode_factor = 0.015
        # Explode the largest slice slightly
        explode = [explode_factor if i == series_data.values.argmax() else 0 for i in range(len(series_data))]

        wedges, texts, autotexts = ax.pie(
            series_data, labels=None, # Labels handled by legend for clarity if many slices
            autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else '', # Show % only for larger slices
            startangle=90, pctdistance=0.80,
            explode=explode, colors=sns.color_palette("Spectral", len(series_data)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.5, 'width':0.6}, # Donut-like effect
            textprops={'fontsize': 8}
        )
        for autotext in autotexts: autotext.set_color('black'); autotext.set_fontweight('normal')
        
        ax.axis('equal')
        # Add a legend, especially if labels were omitted from pie itself
        legend_labels = [f'{label} ({series_data.get(label, 0):,.0f})' if value_col else f'{label} ({series_data.get(label,0)})' for label in series_data.index]

        ax.legend(wedges, legend_labels, title=category_col.replace("_"," ").title(), 
                  loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=8.5, frameon=True, facecolor='#ffffffE0')
        try:
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.70) # Make space for legend
        except: pass
        return fig


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Create varied dummy data for testing
    rng = np.random.RandomState(0)
    n_samples = 150
    sample_df = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_samples, freq='D')),
        'Temperature': rng.normal(20, 5, n_samples) + np.sin(np.linspace(0, 20*np.pi, n_samples))*3,
        'Humidity': rng.uniform(30, 90, n_samples),
        'Pressure': rng.normal(1010, 5, n_samples),
        'WindSpeed': rng.gamma(2, 5, n_samples),
        'Category': rng.choice(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu'], n_samples),
        'Sales': np.abs(rng.normal(100, 40, n_samples)),
        'Group': rng.choice(['G1', 'G2', 'G3', 'G4'], n_samples),
        'Feature1': rng.randn(n_samples),
        'Feature2': rng.randn(n_samples) * 2,
        'Feature3': rng.randn(n_samples) - 1,
        'Feature4': rng.rand(n_samples) * 10,
        'Feature5': rng.choice([10,20,30,40,50], n_samples)
    })
    sample_df.loc[rng.choice(sample_df.index, size=10, replace=False), 'Temperature'] = np.nan # Add some NaNs
    sample_df.loc[rng.choice(sample_df.index, size=5, replace=False), 'Category'] = 'Omega' # Add a less frequent category

    print("--- Testing Enhanced VisualUtils ---")
    plot_functions_to_test = {
        "Scatter Plot": lambda: VisualUtils.create_scatter_plot(sample_df, 'Temperature', 'Humidity', hue='Group'),
        "Box Plot": lambda: VisualUtils.create_box_plot(sample_df, 'Category', 'Sales', hue='Group'),
        "Correlation Matrix": lambda: VisualUtils.create_correlation_matrix(sample_df[['Temperature', 'Humidity', 'Pressure', 'WindSpeed', 'Sales', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']]),
        "Pair Plot": lambda: VisualUtils.create_pairplot(sample_df, hue='Group', max_numeric_cols=4),
        "Distribution Plot": lambda: VisualUtils.create_distribution_plot(sample_df, 'Temperature'),
        "KDE Plot": lambda: VisualUtils.create_kde_plot(sample_df['Humidity'].dropna()),
        "Histogram": lambda: VisualUtils.create_histogram(sample_df['Sales'], bins=15, kde=True),
        "Violin Plot": lambda: VisualUtils.create_violin_plot(sample_df, 'Group', 'WindSpeed', hue='Category'),
        "Line Plot": lambda: VisualUtils.create_line_plot(sample_df, 'Date', 'Temperature', hue='Group'),
        "Area Plot": lambda: VisualUtils.create_area_plot(sample_df, 'Date', 'Pressure', hue='Group'), # Hue will plot multiple lines with fills
        "Bar Plot": lambda: VisualUtils.create_bar_plot(sample_df, 'Category', 'Sales', hue='Group', estimator_func=np.sum),
        "Pie Chart": lambda: VisualUtils.create_pie_chart(sample_df, 'Category', value_col='Sales', max_slices=8),
        "Pie Chart Counts": lambda: VisualUtils.create_pie_chart(sample_df, 'Group')
    }

    figs = []
    for name, func in plot_functions_to_test.items():
        try:
            print(f"Generating: {name}")
            fig = func()
            figs.append(fig)
            # fig.show() # This would open many windows, better to just check plt.show() at end
        except ValueError as ve:
            print(f"ValueError in plotting {name}: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred plotting {name}: {e}")
    
    if figs:
        print(f"\n{len(figs)} plots generated. Call plt.show() to display them if running interactively.")
        # plt.show() # Uncomment to display all plots when running this script directly
    else:
        print("No plots were generated.")