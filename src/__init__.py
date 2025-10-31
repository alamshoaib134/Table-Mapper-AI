# Table Comparison Source Package
__version__ = "1.0.0"
__author__ = "Table Comparison Project"

# Make imports available at package level
from .embeddings import EmbeddingGenerator
from .comparisons import TableComparator
from .utils import load_data, save_results, setup_logging
from .visualizations import create_comparison_plots

__all__ = [
    'EmbeddingGenerator',
    'TableComparator', 
    'load_data',
    'save_results',
    'setup_logging',
    'create_comparison_plots'
]