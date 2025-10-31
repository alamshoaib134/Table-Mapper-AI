"""
Utility Functions Module
Author: Table Comparison Project
Date: October 31, 2025
Description: Common utility functions for data loading, saving, and logging
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from config import *

def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = OUTPUT_DIRS['logs']
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, LOGGING_CONFIG['log_file'])
    
    # Create logger
    logger = logging.getLogger('table_comparison')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOGGING_CONFIG['max_bytes'],
        backupCount=LOGGING_CONFIG['backup_count']
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pandas read functions
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type and load accordingly
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif suffix == '.json':
            return pd.read_json(file_path, **kwargs)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif suffix == '.feather':
            return pd.read_feather(file_path, **kwargs)
        elif suffix in ['.txt', '.tsv']:
            # Try tab-separated values first
            return pd.read_csv(file_path, sep='\t', **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")

def save_results(results: Dict, output_path: str, format: str = 'json') -> None:
    """
    Save results to file in specified format
    
    Args:
        results: Results dictionary to save
        output_path: Output file path
        format: Output format ('json', 'csv', 'excel')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format.lower() == 'csv':
            # Convert matches to DataFrame for CSV export
            if 'matches' in results:
                df = pd.DataFrame(results['matches'])
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("No matches found to save as CSV")
        elif format.lower() == 'excel':
            # Save multiple sheets for different result components
            with pd.ExcelWriter(output_path) as writer:
                if 'matches' in results:
                    pd.DataFrame(results['matches']).to_sheet(writer, 'Matches', index=False)
                
                # Add metadata sheet
                metadata = {
                    'metric': ['total_matches', 'avg_similarity', 'max_similarity'],
                    'value': [
                        results.get('statistics', {}).get('total_matches', 0),
                        results.get('statistics', {}).get('avg_similarity', 0),
                        results.get('statistics', {}).get('max_similarity', 0)
                    ]
                }
                pd.DataFrame(metadata).to_excel(writer, sheet_name='Statistics', index=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
            
    except Exception as e:
        raise ValueError(f"Error saving results to {output_path}: {e}")

def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> bool:
    """
    Validate DataFrame for common issues
    
    Args:
        df: DataFrame to validate
        name: Name for error messages
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if df is None:
        raise ValueError(f"{name} is None")
    
    if df.empty:
        raise ValueError(f"{name} is empty")
    
    if len(df.columns) == 0:
        raise ValueError(f"{name} has no columns")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{name} has duplicate column names: {duplicates}")
    
    return True

def preprocess_dataframe(df: pd.DataFrame, **options) -> pd.DataFrame:
    """
    Preprocess DataFrame with common cleaning operations
    
    Args:
        df: DataFrame to preprocess
        **options: Preprocessing options
            - remove_duplicates: Remove duplicate rows
            - handle_missing: How to handle missing values ('drop', 'fill')
            - fill_value: Value to use for filling missing data
            - normalize_columns: Normalize column names
            
    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    
    # Remove duplicates
    if options.get('remove_duplicates', False):
        df_processed = df_processed.drop_duplicates()
    
    # Handle missing values
    missing_strategy = options.get('handle_missing', 'none')
    if missing_strategy == 'drop':
        df_processed = df_processed.dropna()
    elif missing_strategy == 'fill':
        fill_value = options.get('fill_value', 0)
        df_processed = df_processed.fillna(fill_value)
    
    # Normalize column names
    if options.get('normalize_columns', False):
        df_processed.columns = [normalize_column_name(col) for col in df_processed.columns]
    
    return df_processed

def normalize_column_name(column_name: str) -> str:
    """
    Normalize column name to standard format
    
    Args:
        column_name: Original column name
        
    Returns:
        Normalized column name
    """
    # Convert to lowercase and replace spaces/special chars with underscores
    normalized = column_name.lower()
    normalized = ''.join(c if c.isalnum() else '_' for c in normalized)
    
    # Remove multiple consecutive underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    return normalized

def compute_data_summary(df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive summary of DataFrame
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dictionary containing data summary
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'column_stats': {}
    }
    
    # Per-column statistics
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'unique_count': df[col].nunique(),
            'memory_usage': df[col].memory_usage(deep=True)
        }
        
        # Add type-specific stats
        if np.issubdtype(df[col].dtype, np.number):
            col_stats.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            })
        elif df[col].dtype == 'object':
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                str_lengths = non_null_values.astype(str).str.len()
                col_stats.update({
                    'avg_length': str_lengths.mean(),
                    'min_length': str_lengths.min(),
                    'max_length': str_lengths.max()
                })
        
        summary['column_stats'][col] = col_stats
    
    return summary

def create_comparison_report(results: Dict, output_dir: str = None) -> str:
    """
    Create a comprehensive comparison report
    
    Args:
        results: Comparison results dictionary
        output_dir: Directory to save report
        
    Returns:
        Path to generated report
    """
    if output_dir is None:
        output_dir = OUTPUT_DIRS['results']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"comparison_report_{timestamp}.md")
    
    # Generate report content
    report_content = generate_markdown_report(results)
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    return report_path

def generate_markdown_report(results: Dict) -> str:
    """
    Generate markdown report from comparison results
    
    Args:
        results: Comparison results dictionary
        
    Returns:
        Markdown report content
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Table Comparison Report

**Generated:** {timestamp}
**Version:** {VERSION}

## Overview

This report contains the results of comparing two tables using the Table Comparison toolkit.

### Table Information

- **Table 1:** {results.get('table1_shape', 'Unknown')} (rows × columns)
- **Table 2:** {results.get('table2_shape', 'Unknown')} (rows × columns)
- **Methods Used:** {', '.join(results.get('methods_used', []))}

### Summary Statistics

"""
    
    # Add method-specific results
    for method, method_results in results.get('comparisons', {}).items():
        stats = method_results.get('statistics', {})
        
        report += f"""
#### {method.title()} Method

- **Total Matches:** {stats.get('total_matches', 0)}
- **Average Similarity:** {stats.get('avg_similarity', 0):.3f}
- **Max Similarity:** {stats.get('max_similarity', 0):.3f}
- **Min Similarity:** {stats.get('min_similarity', 0):.3f}

"""
        
        # Add top matches
        matches = method_results.get('matches', [])
        if matches:
            report += "**Top Matches:**\n\n"
            report += "| Table 1 Column | Table 2 Column | Similarity | Confidence |\n"
            report += "|----------------|----------------|------------|------------|\n"
            
            for match in matches[:10]:  # Top 10 matches
                report += f"| {match['table1_column']} | {match['table2_column']} | {match['similarity']:.3f} | {match.get('confidence', 'N/A')} |\n"
            
            report += "\n"
    
    report += """
## Methodology

This comparison used the following approaches:

1. **Semantic Embeddings:** Uses Ollama embedding models to capture semantic meaning
2. **Fuzzy String Matching:** Compares column names using string similarity algorithms
3. **Statistical Signatures:** Compares columns based on statistical properties
4. **Content Analysis:** Analyzes actual data content to find similar columns
5. **Ensemble Method:** Combines multiple approaches for robust matching

For more details, see the project documentation.
"""
    
    return report

def check_system_requirements() -> Dict[str, bool]:
    """
    Check if system requirements are met
    
    Returns:
        Dictionary of requirement checks
    """
    requirements = {}
    
    # Check Python packages
    try:
        import pandas
        requirements['pandas'] = True
    except ImportError:
        requirements['pandas'] = False
    
    try:
        import numpy
        requirements['numpy'] = True
    except ImportError:
        requirements['numpy'] = False
    
    try:
        import requests
        requirements['requests'] = True
    except ImportError:
        requirements['requests'] = False
    
    try:
        import sklearn
        requirements['sklearn'] = True
    except ImportError:
        requirements['sklearn'] = False
    
    # Check Ollama connection
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        requirements['ollama'] = response.status_code == 200
    except:
        requirements['ollama'] = False
    
    return requirements

def print_system_status():
    """Print system status and requirements"""
    requirements = check_system_requirements()
    
    print("System Requirements Check:")
    print("=" * 50)
    
    for req, status in requirements.items():
        status_str = "✓ OK" if status else "✗ MISSING"
        print(f"{req:20} {status_str}")
    
    if not all(requirements.values()):
        print("\nSome requirements are missing. Please install them using:")
        print("pip install -r requirements.txt")
        
        if not requirements.get('ollama', False):
            print("\nOllama is not running. Please start Ollama and install the required models:")
            print("ollama run mxbai-embed-large")

if __name__ == "__main__":
    print_system_status()