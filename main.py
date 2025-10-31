#!/usr/bin/env python3
"""
Table Comparison Command Line Interface
Author: Table Comparison Project
Date: October 31, 2025
Description: CLI tool for running table comparisons and generating analysis reports
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Local imports (assuming modular structure exists)
try:
    from config import *
    from src.embeddings import EmbeddingGenerator
    from src.comparisons import TableComparator
    from src.utils import load_data, save_results, setup_logging
    from src.visualizations import create_comparison_plots
except ImportError:
    print("Warning: Modular structure not found. Using notebook-style imports.")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    from config import *

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Intelligent Table Comparison Tool with Advanced Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison of two CSV files
  python main.py --table1 data/table1.csv --table2 data/table2.csv
  
  # Use specific embedding model
  python main.py --table1 data/table1.csv --table2 data/table2.csv --model nomic-embed-text
  
  # Generate comprehensive report with visualizations
  python main.py --table1 data/table1.csv --table2 data/table2.csv --output report.json --plots
  
  # Use ensemble method with custom weights
  python main.py --table1 data/table1.csv --table2 data/table2.csv --method ensemble --config semantic_focused
  
  # Quick comparison with specific threshold
  python main.py --table1 data/table1.csv --table2 data/table2.csv --threshold 0.8 --quick
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--table1', 
        type=str, 
        required=True,
        help='Path to first CSV table file'
    )
    
    parser.add_argument(
        '--table2', 
        type=str, 
        required=True,
        help='Path to second CSV table file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--method',
        type=str,
        choices=['semantic', 'fuzzy', 'statistical', 'content', 'hybrid', 'ensemble'],
        default='ensemble',
        help='Comparison method to use (default: ensemble)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=list(EMBEDDING_MODELS.keys()),
        default=DEFAULT_EMBEDDING_MODEL,
        help=f'Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=SIMILARITY_THRESHOLDS['high_confidence'],
        help=f'Similarity threshold for matches (default: {SIMILARITY_THRESHOLDS["high_confidence"]})'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        choices=list(WEIGHT_CONFIGS.keys()),
        default='balanced',
        help='Weight configuration for ensemble method (default: balanced)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for results (JSON format)'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: fewer samples, faster processing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=SAMPLE_SIZE,
        help=f'Number of samples for content analysis (default: {SAMPLE_SIZE})'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available embedding models and exit'
    )
    
    return parser.parse_args()

def list_available_models() -> None:
    """Display available embedding models and their specifications"""
    print("\nAvailable Embedding Models:")
    print("=" * 50)
    
    for model_name, specs in EMBEDDING_MODELS.items():
        print(f"\n{model_name}:")
        print(f"  Dimensions: {specs['dimensions']}")
        print(f"  Best for: {specs['best_for']}")
        print(f"  Speed: {specs['speed']}")
        print(f"  Size: {specs['size']}")
    
    print(f"\nDefault model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Fallback models: {', '.join(FALLBACK_MODELS)}")

def validate_inputs(args: argparse.Namespace) -> bool:
    """Validate input arguments and files"""
    # Check if files exist
    for table_path in [args.table1, args.table2]:
        if not Path(table_path).exists():
            print(f"Error: File not found: {table_path}")
            return False
        
        if not table_path.endswith('.csv'):
            print(f"Warning: {table_path} is not a CSV file. Attempting to read anyway...")
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        return False
    
    # Validate sample size
    if args.sample_size <= 0:
        print(f"Error: Sample size must be positive, got {args.sample_size}")
        return False
    
    return True

def load_tables(table1_path: str, table2_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load tables from CSV files"""
    try:
        print(f"Loading {table1_path}...")
        table1 = pd.read_csv(table1_path)
        print(f"  Shape: {table1.shape}")
        
        print(f"Loading {table2_path}...")
        table2 = pd.read_csv(table2_path)
        print(f"  Shape: {table2.shape}")
        
        return table1, table2
    
    except Exception as e:
        print(f"Error loading tables: {e}")
        sys.exit(1)

def run_comparison(
    table1: pd.DataFrame, 
    table2: pd.DataFrame, 
    args: argparse.Namespace
) -> Dict:
    """Run the specified comparison method"""
    
    print(f"\nRunning {args.method} comparison...")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    
    # Adjust sample size for quick mode
    sample_size = args.sample_size // 2 if args.quick else args.sample_size
    
    # This is a placeholder for the actual comparison logic
    # In a full implementation, this would use the modular components
    results = {
        'method': args.method,
        'model': args.model,
        'threshold': args.threshold,
        'sample_size': sample_size,
        'table1_columns': list(table1.columns),
        'table2_columns': list(table2.columns),
        'table1_shape': table1.shape,
        'table2_shape': table2.shape,
        'matches': [],
        'statistics': {},
        'performance': {}
    }
    
    # Basic column name similarity (placeholder)
    print("Computing column similarities...")
    column_matches = []
    
    for i, col1 in enumerate(table1.columns):
        for j, col2 in enumerate(table2.columns):
            # Simple string similarity as placeholder
            similarity = len(set(col1.lower()) & set(col2.lower())) / len(set(col1.lower()) | set(col2.lower()))
            
            if similarity >= args.threshold:
                match = {
                    'table1_column': col1,
                    'table2_column': col2,
                    'similarity': similarity,
                    'method': 'basic_string'
                }
                column_matches.append(match)
    
    results['matches'] = column_matches
    results['statistics']['total_matches'] = len(column_matches)
    results['statistics']['avg_similarity'] = sum(m['similarity'] for m in column_matches) / max(len(column_matches), 1)
    
    print(f"Found {len(column_matches)} matches above threshold {args.threshold}")
    
    return results

def display_results(results: Dict) -> None:
    """Display comparison results in a formatted way"""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nMethod: {results['method']}")
    print(f"Model: {results['model']}")
    print(f"Threshold: {results['threshold']}")
    
    print(f"\nTable 1: {results['table1_shape'][1]} columns, {results['table1_shape'][0]} rows")
    print(f"Table 2: {results['table2_shape'][1]} columns, {results['table2_shape'][0]} rows")
    
    print(f"\nMatches Found: {results['statistics']['total_matches']}")
    print(f"Average Similarity: {results['statistics']['avg_similarity']:.3f}")
    
    if results['matches']:
        print("\nTop Matches:")
        print("-" * 60)
        sorted_matches = sorted(results['matches'], key=lambda x: x['similarity'], reverse=True)
        
        for match in sorted_matches[:10]:  # Show top 10
            print(f"{match['table1_column']} ↔ {match['table2_column']} ({match['similarity']:.3f})")
    
    print("\n" + "="*80)

def save_output(results: Dict, output_path: str) -> None:
    """Save results to JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main CLI function"""
    args = parse_arguments()
    
    # Handle special commands
    if args.list_models:
        list_available_models()
        return
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    # setup_logging(log_level)  # Uncomment when modular structure exists
    
    # Validate inputs
    if not validate_inputs(args):
        sys.exit(1)
    
    # Load tables
    table1, table2 = load_tables(args.table1, args.table2)
    
    # Run comparison
    results = run_comparison(table1, table2, args)
    
    # Display results
    display_results(results)
    
    # Save output if specified
    if args.output:
        save_output(results, args.output)
    
    # Generate plots if requested
    if args.plots:
        print("\nPlot generation requested but not implemented in CLI mode.")
        print("Please use the Jupyter notebook for comprehensive visualizations.")

if __name__ == "__main__":
    main()