"""
Visualization Module
Author: Table Comparison Project
Date: October 31, 2025
Description: Creates visualizations for table comparison results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os

from config import *

def create_comparison_plots(results: Dict, output_dir: str = None) -> List[str]:
    """
    Create comprehensive visualization plots for comparison results
    
    Args:
        results: Comparison results dictionary
        output_dir: Directory to save plots
        
    Returns:
        List of paths to saved plot files
    """
    if output_dir is None:
        output_dir = OUTPUT_DIRS['figures']
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = []
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    try:
        # 1. Similarity Matrix Heatmap
        for method, method_results in results.get('comparisons', {}).items():
            if 'similarity_matrix' in method_results:
                plot_path = create_similarity_heatmap(
                    method_results, 
                    results.get('table1_columns', []),
                    results.get('table2_columns', []),
                    method,
                    output_dir
                )
                if plot_path:
                    saved_plots.append(plot_path)
        
        # 2. Method Comparison Plot
        plot_path = create_method_comparison_plot(results, output_dir)
        if plot_path:
            saved_plots.append(plot_path)
        
        # 3. Confidence Distribution Plot
        plot_path = create_confidence_distribution_plot(results, output_dir)
        if plot_path:
            saved_plots.append(plot_path)
        
        # 4. Match Quality Analysis
        plot_path = create_match_quality_analysis(results, output_dir)
        if plot_path:
            saved_plots.append(plot_path)
        
        # 5. Performance Overview
        plot_path = create_performance_overview(results, output_dir)
        if plot_path:
            saved_plots.append(plot_path)
    
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return saved_plots

def create_similarity_heatmap(
    method_results: Dict,
    table1_cols: List[str],
    table2_cols: List[str],
    method_name: str,
    output_dir: str
) -> Optional[str]:
    """Create similarity matrix heatmap"""
    try:
        similarity_matrix = np.array(method_results['similarity_matrix'])
        
        # Create figure
        plt.figure(figsize=VISUALIZATION_CONFIG['heatmap']['figsize'])
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            xticklabels=table2_cols,
            yticklabels=table1_cols,
            cmap=VISUALIZATION_CONFIG['heatmap']['cmap'],
            annot=True,
            fmt=VISUALIZATION_CONFIG['heatmap']['fmt'],
            cbar_kws={'label': 'Similarity Score'},
            square=False
        )
        
        plt.title(f'{method_name.title()} Method - Column Similarity Matrix', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Table 2 Columns', fontsize=12, fontweight='bold')
        plt.ylabel('Table 1 Columns', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'similarity_heatmap_{method_name}.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating similarity heatmap for {method_name}: {e}")
        plt.close()
        return None

def create_method_comparison_plot(results: Dict, output_dir: str) -> Optional[str]:
    """Create method comparison bar plot"""
    try:
        methods = []
        match_counts = []
        avg_similarities = []
        
        # Extract data for each method
        for method, method_results in results.get('comparisons', {}).items():
            stats = method_results.get('statistics', {})
            methods.append(method.title())
            match_counts.append(stats.get('total_matches', 0))
            avg_similarities.append(stats.get('avg_similarity', 0))
        
        if not methods:
            return None
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VISUALIZATION_CONFIG['comparison_charts']['figsize'])
        
        # Plot 1: Match counts
        bars1 = ax1.bar(methods, match_counts, color=COLOR_PALETTE[:len(methods)], 
                       alpha=VISUALIZATION_CONFIG['comparison_charts']['alpha'])
        ax1.set_title('Number of Matches by Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Matches', fontweight='bold')
        ax1.grid(True, alpha=VISUALIZATION_CONFIG['comparison_charts']['grid_alpha'])
        
        # Add value labels on bars
        for bar, count in zip(bars1, match_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Average similarities
        bars2 = ax2.bar(methods, avg_similarities, color=COLOR_PALETTE[:len(methods)], 
                       alpha=VISUALIZATION_CONFIG['comparison_charts']['alpha'])
        ax2.set_title('Average Similarity by Method', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Similarity', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=VISUALIZATION_CONFIG['comparison_charts']['grid_alpha'])
        
        # Add value labels on bars
        for bar, sim in zip(bars2, avg_similarities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{sim:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Method Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'method_comparison.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating method comparison plot: {e}")
        plt.close()
        return None

def create_confidence_distribution_plot(results: Dict, output_dir: str) -> Optional[str]:
    """Create confidence distribution plot"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        axes = axes.flatten()
        
        method_idx = 0
        for method, method_results in results.get('comparisons', {}).items():
            if method_idx >= 4:  # Limit to 4 methods for layout
                break
                
            stats = method_results.get('statistics', {})
            conf_dist = stats.get('confidence_distribution', {})
            
            if conf_dist:
                # Create pie chart for confidence distribution
                labels = list(conf_dist.keys())
                sizes = list(conf_dist.values())
                colors = VISUALIZATION_CONFIG['pie_charts']['colors'][:len(labels)]
                
                axes[method_idx].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                   startangle=VISUALIZATION_CONFIG['pie_charts']['startangle'])
                axes[method_idx].set_title(f'{method.title()} Method\nConfidence Distribution', 
                                         fontweight='bold')
            else:
                axes[method_idx].text(0.5, 0.5, 'No data', ha='center', va='center',
                                    transform=axes[method_idx].transAxes)
                axes[method_idx].set_title(f'{method.title()} Method', fontweight='bold')
            
            method_idx += 1
        
        # Hide unused subplots
        for idx in range(method_idx, 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Match Confidence Distribution by Method', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'confidence_distribution.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating confidence distribution plot: {e}")
        plt.close()
        return None

def create_match_quality_analysis(results: Dict, output_dir: str) -> Optional[str]:
    """Create match quality analysis plot"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        
        # Collect all matches across methods
        all_matches = []
        method_labels = []
        
        for method, method_results in results.get('comparisons', {}).items():
            matches = method_results.get('matches', [])
            similarities = [match['similarity'] for match in matches]
            all_matches.extend(similarities)
            method_labels.extend([method.title()] * len(similarities))
        
        if all_matches:
            # Plot 1: Similarity distribution histogram
            ax1.hist(all_matches, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Distribution of Similarity Scores', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Similarity Score', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add threshold lines
            for name, threshold in SIMILARITY_THRESHOLDS.items():
                if threshold <= 1.0:  # Only plot meaningful thresholds
                    ax1.axvline(threshold, color='red', linestyle='--', alpha=0.7,
                              label=f'{name}: {threshold}')
            ax1.legend()
            
            # Plot 2: Box plot by method
            if len(set(method_labels)) > 1:
                match_df = pd.DataFrame({
                    'similarity': all_matches,
                    'method': method_labels
                })
                sns.boxplot(data=match_df, x='method', y='similarity', ax=ax2)
                ax2.set_title('Similarity Score Distribution by Method', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Method', fontweight='bold')
                ax2.set_ylabel('Similarity Score', fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'Single method analysis', ha='center', va='center',
                        transform=ax2.transAxes)
        else:
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'No matches found', ha='center', va='center',
                       transform=ax.transAxes)
        
        plt.suptitle('Match Quality Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'match_quality_analysis.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating match quality analysis: {e}")
        plt.close()
        return None

def create_performance_overview(results: Dict, output_dir: str) -> Optional[str]:
    """Create performance overview dashboard"""
    try:
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Method performance summary (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        
        methods = []
        metrics = {
            'Total Matches': [],
            'Avg Similarity': [],
            'Max Similarity': []
        }
        
        for method, method_results in results.get('comparisons', {}).items():
            stats = method_results.get('statistics', {})
            methods.append(method.title())
            metrics['Total Matches'].append(stats.get('total_matches', 0))
            metrics['Avg Similarity'].append(stats.get('avg_similarity', 0))
            metrics['Max Similarity'].append(stats.get('max_similarity', 0))
        
        if methods:
            x = np.arange(len(methods))
            width = 0.25
            
            # Normalize metrics for comparison
            max_matches = max(metrics['Total Matches']) if metrics['Total Matches'] else 1
            normalized_matches = [m / max_matches for m in metrics['Total Matches']]
            
            ax1.bar(x - width, normalized_matches, width, label='Matches (normalized)', alpha=0.8)
            ax1.bar(x, metrics['Avg Similarity'], width, label='Avg Similarity', alpha=0.8)
            ax1.bar(x + width, metrics['Max Similarity'], width, label='Max Similarity', alpha=0.8)
            
            ax1.set_title('Method Performance Overview', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Methods', fontweight='bold')
            ax1.set_ylabel('Score', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Top matches table (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('tight')
        ax2.axis('off')
        
        # Get top matches from best performing method
        best_method = None
        best_avg_sim = 0
        for method, method_results in results.get('comparisons', {}).items():
            avg_sim = method_results.get('statistics', {}).get('avg_similarity', 0)
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_method = method_results
        
        if best_method and best_method.get('matches'):
            top_matches = best_method['matches'][:5]  # Top 5 matches
            table_data = []
            for match in top_matches:
                table_data.append([
                    match['table1_column'][:15] + '...' if len(match['table1_column']) > 15 else match['table1_column'],
                    match['table2_column'][:15] + '...' if len(match['table2_column']) > 15 else match['table2_column'],
                    f"{match['similarity']:.3f}"
                ])
            
            table = ax2.table(cellText=table_data,
                            colLabels=['Table 1', 'Table 2', 'Similarity'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax2.set_title('Top 5 Matches', fontweight='bold', pad=20)
        
        # 3. Summary statistics (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        
        total_table1_cols = len(results.get('table1_columns', []))
        total_table2_cols = len(results.get('table2_columns', []))
        
        # Calculate overall statistics
        all_matches_count = 0
        all_avg_similarity = 0
        method_count = 0
        
        for method, method_results in results.get('comparisons', {}).items():
            stats = method_results.get('statistics', {})
            all_matches_count += stats.get('total_matches', 0)
            all_avg_similarity += stats.get('avg_similarity', 0)
            method_count += 1
        
        overall_avg_similarity = all_avg_similarity / method_count if method_count > 0 else 0
        
        summary_text = f"""
Summary Statistics

Table 1 Columns: {total_table1_cols}
Table 2 Columns: {total_table2_cols}

Total Matches Found: {all_matches_count}
Methods Tested: {method_count}
Overall Avg Similarity: {overall_avg_similarity:.3f}

Coverage:
Table 1: {(all_matches_count/total_table1_cols*100) if total_table1_cols > 0 else 0:.1f}%
Table 2: {(all_matches_count/total_table2_cols*100) if total_table2_cols > 0 else 0:.1f}%
        """
        
        ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        ax3.axis('off')
        
        # 4. Method comparison radar (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        
        if len(methods) > 1:
            # Create simple comparison chart
            method_scores = []
            for method in methods:
                method_result = results['comparisons'][method.lower()]
                stats = method_result.get('statistics', {})
                
                # Normalize scores
                match_score = min(stats.get('total_matches', 0) / 10, 1.0)  # Cap at 10 matches
                sim_score = stats.get('avg_similarity', 0)
                
                method_scores.append((match_score + sim_score) / 2)
            
            ax4.bar(methods, method_scores, color=COLOR_PALETTE[:len(methods)], alpha=0.7)
            ax4.set_title('Method Overall Score', fontweight='bold')
            ax4.set_ylabel('Combined Score', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'Single method\nanalysis', ha='center', va='center',
                    transform=ax4.transAxes, fontweight='bold')
            ax4.axis('off')
        
        # 5. Threshold analysis (bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Analyze matches at different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_counts = {method: [] for method in methods}
        
        for threshold in thresholds:
            for method in methods:
                method_key = method.lower()
                if method_key in results.get('comparisons', {}):
                    matches = results['comparisons'][method_key].get('matches', [])
                    count = sum(1 for match in matches if match['similarity'] >= threshold)
                    threshold_counts[method].append(count)
                else:
                    threshold_counts[method].append(0)
        
        for i, method in enumerate(methods):
            ax5.plot(thresholds, threshold_counts[method], marker='o', 
                    label=method, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
        
        ax5.set_title('Matches at Different Similarity Thresholds', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Similarity Threshold', fontweight='bold')
        ax5.set_ylabel('Number of Matches', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Table Comparison Performance Dashboard', fontsize=18, fontweight='bold')
        
        # Save plot
        plot_path = os.path.join(output_dir, 'performance_overview.png')
        plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating performance overview: {e}")
        plt.close()
        return None

def save_comparison_summary(results: Dict, output_path: str) -> None:
    """Save a summary of comparison results to CSV"""
    try:
        summary_data = []
        
        for method, method_results in results.get('comparisons', {}).items():
            stats = method_results.get('statistics', {})
            
            summary_data.append({
                'Method': method.title(),
                'Total_Matches': stats.get('total_matches', 0),
                'Avg_Similarity': stats.get('avg_similarity', 0),
                'Max_Similarity': stats.get('max_similarity', 0),
                'Min_Similarity': stats.get('min_similarity', 0)
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"Summary saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving summary: {e}")

if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully")
    print("Use create_comparison_plots(results) to generate visualizations")