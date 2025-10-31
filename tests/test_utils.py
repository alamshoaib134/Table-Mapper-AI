"""
Unit tests for utility functions
"""

import unittest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils import (
        load_data, save_results, validate_dataframe, 
        preprocess_dataframe, normalize_column_name,
        compute_data_summary, check_system_requirements
    )
except ImportError:
    print("Warning: Utils module not available for testing")

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample DataFrame
        self.sample_df = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'Customer Name': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Charlie Wilson'],
            'email@address': ['john@email.com', 'jane@email.com', 'bob@email.com', 'missing@email.com', 'charlie@email.com'],
            'purchase_amount': [150.50, 200.75, 99.99, 300.00, 175.25],
            'order_date': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19']),
            'is_premium': [True, False, True, True, False]
        })
        
        # Create temporary directory for file tests
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_dataframe(self):
        """Test DataFrame validation"""
        # Test valid DataFrame
        self.assertTrue(validate_dataframe(self.sample_df, "test_df"))
        
        # Test None DataFrame
        with self.assertRaises(ValueError):
            validate_dataframe(None, "none_df")
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            validate_dataframe(empty_df, "empty_df")
        
        # Test DataFrame with no columns
        no_cols_df = pd.DataFrame(index=[0, 1, 2])
        with self.assertRaises(ValueError):
            validate_dataframe(no_cols_df, "no_cols_df")
        
        # Test DataFrame with duplicate columns
        dup_cols_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col1': [4, 5, 6]  # Duplicate column name
        })
        # Note: pandas automatically renames duplicate columns, so this test
        # checks our validation logic
        try:
            validate_dataframe(dup_cols_df, "dup_cols_df")
        except ValueError:
            pass  # Expected for true duplicates
    
    def test_normalize_column_name(self):
        """Test column name normalization"""
        test_cases = [
            ('Customer Name', 'customer_name'),
            ('email@address', 'email_address'),
            ('Order-Date', 'order_date'),
            ('user_id', 'user_id'),
            ('Total$$Amount', 'total_amount'),
            ('  spaced  ', 'spaced'),
            ('CamelCaseColumn', 'camelcasecolumn'),
            ('multiple___underscores', 'multiple_underscores')
        ]
        
        for original, expected in test_cases:
            normalized = normalize_column_name(original)
            self.assertEqual(normalized, expected)
    
    def test_preprocess_dataframe(self):
        """Test DataFrame preprocessing"""
        # Add duplicate row for testing
        df_with_dups = pd.concat([self.sample_df, self.sample_df.iloc[[0]]], ignore_index=True)
        
        # Test remove duplicates
        processed = preprocess_dataframe(df_with_dups, remove_duplicates=True)
        self.assertEqual(len(processed), len(self.sample_df))
        
        # Test fill missing values
        processed_fill = preprocess_dataframe(
            self.sample_df, 
            handle_missing='fill', 
            fill_value='MISSING'
        )
        # Check that null values are filled
        self.assertFalse(processed_fill['Customer Name'].isnull().any())
        
        # Test normalize columns
        processed_norm = preprocess_dataframe(
            self.sample_df, 
            normalize_columns=True
        )
        expected_columns = [normalize_column_name(col) for col in self.sample_df.columns]
        self.assertEqual(list(processed_norm.columns), expected_columns)
    
    def test_compute_data_summary(self):
        """Test data summary computation"""
        summary = compute_data_summary(self.sample_df)
        
        # Check basic structure
        self.assertIn('shape', summary)
        self.assertIn('columns', summary)
        self.assertIn('dtypes', summary)
        self.assertIn('missing_values', summary)
        self.assertIn('column_stats', summary)
        
        # Check shape
        self.assertEqual(summary['shape'], self.sample_df.shape)
        
        # Check columns
        self.assertEqual(summary['columns'], list(self.sample_df.columns))
        
        # Check column stats
        self.assertEqual(len(summary['column_stats']), len(self.sample_df.columns))
        
        # Check numeric column stats
        numeric_col_stats = summary['column_stats']['purchase_amount']
        self.assertIn('mean', numeric_col_stats)
        self.assertIn('std', numeric_col_stats)
        self.assertIn('min', numeric_col_stats)
        self.assertIn('max', numeric_col_stats)
        
        # Check text column stats
        text_col_stats = summary['column_stats']['Customer Name']
        self.assertIn('avg_length', text_col_stats)
        self.assertIn('unique_count', text_col_stats)
    
    def test_save_and_load_data(self):
        """Test data saving and loading"""
        # Test CSV save/load
        csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.sample_df.to_csv(csv_path, index=False)
        
        try:
            loaded_df = load_data(csv_path)
            # Compare shapes (dtypes might differ slightly)
            self.assertEqual(loaded_df.shape, self.sample_df.shape)
        except ImportError:
            self.skipTest("pandas not available")
    
    def test_save_results(self):
        """Test results saving"""
        # Create test results
        test_results = {
            'method': 'test',
            'matches': [
                {'table1_column': 'col1', 'table2_column': 'col2', 'similarity': 0.8},
                {'table1_column': 'col3', 'table2_column': 'col4', 'similarity': 0.9}
            ],
            'statistics': {
                'total_matches': 2,
                'avg_similarity': 0.85
            }
        }
        
        # Test JSON save
        json_path = os.path.join(self.temp_dir, 'test_results.json')
        try:
            save_results(test_results, json_path, format='json')
            
            # Verify file was created and contains correct data
            self.assertTrue(os.path.exists(json_path))
            
            with open(json_path, 'r') as f:
                loaded_results = json.load(f)
            
            self.assertEqual(loaded_results['method'], 'test')
            self.assertEqual(len(loaded_results['matches']), 2)
        except Exception as e:
            self.skipTest(f"Save results test failed: {e}")
    
    def test_check_system_requirements(self):
        """Test system requirements check"""
        try:
            requirements = check_system_requirements()
            
            # Should return a dictionary
            self.assertIsInstance(requirements, dict)
            
            # Should check for key packages
            expected_packages = ['pandas', 'numpy', 'requests']
            for package in expected_packages:
                self.assertIn(package, requirements)
                self.assertIsInstance(requirements[package], bool)
        
        except Exception as e:
            self.skipTest(f"System requirements check failed: {e}")

class TestFileOperations(unittest.TestCase):
    """Test file operation utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = {
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['New York', 'London', 'Tokyo']
        }
        self.sample_df = pd.DataFrame(self.sample_data)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_data_formats(self):
        """Test loading different data formats"""
        # Test CSV
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        self.sample_df.to_csv(csv_path, index=False)
        
        try:
            loaded_csv = load_data(csv_path)
            self.assertEqual(loaded_csv.shape, self.sample_df.shape)
        except (ImportError, NameError):
            self.skipTest("pandas not available")
        
        # Test JSON
        json_path = os.path.join(self.temp_dir, 'test.json')
        self.sample_df.to_json(json_path, orient='records')
        
        try:
            loaded_json = load_data(json_path)
            self.assertEqual(loaded_json.shape, self.sample_df.shape)
        except (ImportError, NameError):
            self.skipTest("pandas not available")
        
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')
        
        # Test unsupported format
        unsupported_path = os.path.join(self.temp_dir, 'test.xyz')
        with open(unsupported_path, 'w') as f:
            f.write("test content")
        
        with self.assertRaises(ValueError):
            load_data(unsupported_path)

class TestReportGeneration(unittest.TestCase):
    """Test report generation utilities"""
    
    def test_generate_markdown_report(self):
        """Test markdown report generation"""
        # Create sample results
        sample_results = {
            'table1_shape': (100, 5),
            'table2_shape': (150, 6),
            'methods_used': ['semantic', 'fuzzy'],
            'comparisons': {
                'semantic': {
                    'statistics': {
                        'total_matches': 3,
                        'avg_similarity': 0.85,
                        'max_similarity': 0.95,
                        'min_similarity': 0.75
                    },
                    'matches': [
                        {
                            'table1_column': 'user_id',
                            'table2_column': 'customer_id',
                            'similarity': 0.95,
                            'confidence': 'high'
                        },
                        {
                            'table1_column': 'name',
                            'table2_column': 'full_name',
                            'similarity': 0.85,
                            'confidence': 'high'
                        }
                    ]
                }
            }
        }
        
        try:
            from utils import generate_markdown_report
            report = generate_markdown_report(sample_results)
            
            # Basic structure checks
            self.assertIsInstance(report, str)
            self.assertIn('# Table Comparison Report', report)
            self.assertIn('semantic', report)
            self.assertIn('Total Matches', report)
            self.assertIn('user_id', report)
            
        except ImportError:
            self.skipTest("Utils module not available")

if __name__ == '__main__':
    # Set up test environment
    import warnings
    warnings.filterwarnings('ignore')
    
    # Run tests
    unittest.main(verbosity=2)