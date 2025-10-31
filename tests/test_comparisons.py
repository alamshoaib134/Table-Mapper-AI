"""
Unit tests for comparison functionality
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from comparisons import TableComparator
    from config import SIMILARITY_THRESHOLDS
except ImportError:
    # Graceful handling if dependencies are not available
    print("Warning: Some dependencies not available for testing")

class TestTableComparator(unittest.TestCase):
    """Test cases for TableComparator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample DataFrames for testing
        self.table1 = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'full_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'email_addr': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
            'purchase_total': [150.50, 200.75, 99.99, 300.00, 175.25],
            'created_date': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19']),
            'premium_user': [True, False, True, True, False]
        })
        
        self.table2 = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'customer_name': ['Alice Brown', 'Charlie Green', 'Diana White', 'Edward Black', 'Fiona Red'],
            'email_address': ['alice@demo.com', 'charlie@demo.com', 'diana@demo.com', 'edward@demo.com', 'fiona@demo.com'],
            'order_amount': [120.0, 180.0, 90.0, 250.0, 160.0],
            'order_date': pd.to_datetime(['2023-02-01', '2023-02-02', '2023-02-03', '2023-02-04', '2023-02-05']),
            'is_premium': [True, True, False, True, False]
        })
        
        self.comparator = TableComparator()
    
    def test_initialization(self):
        """Test TableComparator initialization"""
        # Test default initialization
        comp = TableComparator()
        self.assertIsNotNone(comp.embedding_generator)
        
        # Test custom model initialization
        comp_custom = TableComparator('nomic-embed-text')
        self.assertEqual(comp_custom.embedding_generator.model_name, 'nomic-embed-text')
    
    def test_compute_similarity_matrix(self):
        """Test similarity matrix computation"""
        # Create mock embeddings
        emb1 = {
            'col1': np.array([1, 0, 0]),
            'col2': np.array([0, 1, 0])
        }
        emb2 = {
            'col3': np.array([1, 0, 0]),  # Should be similar to col1
            'col4': np.array([0, 0, 1])   # Should be different from both
        }
        
        # Compute similarity matrix
        similarity_matrix = self.comparator._compute_similarity_matrix(emb1, emb2)
        
        # Assertions
        self.assertEqual(similarity_matrix.shape, (2, 2))
        self.assertGreater(similarity_matrix[0, 0], similarity_matrix[0, 1])  # col1 more similar to col3
    
    def test_extract_matches(self):
        """Test match extraction from similarity matrix"""
        # Create mock similarity matrix
        similarity_matrix = np.array([
            [0.9, 0.2],  # col1 matches col3 strongly
            [0.3, 0.8]   # col2 matches col4 strongly
        ])
        
        cols1 = ['col1', 'col2']
        cols2 = ['col3', 'col4']
        threshold = 0.7
        
        matches = self.comparator._extract_matches(similarity_matrix, cols1, cols2, threshold)
        
        # Assertions
        self.assertEqual(len(matches), 2)  # Two matches above threshold
        self.assertEqual(matches[0]['table1_column'], 'col1')
        self.assertEqual(matches[0]['table2_column'], 'col3')
        self.assertAlmostEqual(matches[0]['similarity'], 0.9)
    
    def test_compute_match_statistics(self):
        """Test match statistics computation"""
        # Create mock matches
        matches = [
            {'similarity': 0.9, 'confidence': 'high'},
            {'similarity': 0.8, 'confidence': 'high'},
            {'similarity': 0.7, 'confidence': 'medium'},
            {'similarity': 0.6, 'confidence': 'medium'}
        ]
        
        stats = self.comparator._compute_match_statistics(matches)
        
        # Assertions
        self.assertEqual(stats['total_matches'], 4)
        self.assertAlmostEqual(stats['avg_similarity'], 0.75)
        self.assertEqual(stats['max_similarity'], 0.9)
        self.assertEqual(stats['min_similarity'], 0.6)
        self.assertEqual(stats['confidence_distribution']['high'], 2)
        self.assertEqual(stats['confidence_distribution']['medium'], 2)
    
    def test_classify_confidence(self):
        """Test confidence level classification"""
        # Test different similarity scores
        test_cases = [
            (0.95, 'very_high'),
            (0.85, 'very_high'),
            (0.75, 'high'),
            (0.65, 'high'),
            (0.55, 'medium'),
            (0.45, 'medium'),
            (0.35, 'low'),
            (0.25, 'low'),
            (0.15, 'very_low')
        ]
        
        for similarity, expected_confidence in test_cases:
            confidence = self.comparator._classify_confidence(similarity)
            self.assertEqual(confidence, expected_confidence)
    
    def test_generate_statistical_signatures(self):
        """Test statistical signature generation"""
        signatures = self.comparator._generate_statistical_signatures(self.table1)
        
        # Check that signatures are generated for all columns
        self.assertEqual(len(signatures), len(self.table1.columns))
        
        # Check numeric column signature
        numeric_sig = signatures['purchase_total']
        self.assertIn('mean', numeric_sig)
        self.assertIn('std', numeric_sig)
        self.assertIn('min', numeric_sig)
        self.assertIn('max', numeric_sig)
        
        # Check text column signature
        text_sig = signatures['full_name']
        self.assertIn('avg_length', text_sig)
        self.assertIn('unique_count', text_sig)
        
        # Check boolean column signature
        bool_sig = signatures['premium_user']
        self.assertEqual(bool_sig['dtype'], 'bool')
    
    def test_compare_signatures(self):
        """Test signature comparison"""
        # Create similar signatures
        sig1 = {
            'dtype': 'float64',
            'mean': 100.0,
            'std': 10.0,
            'unique_count': 50
        }
        
        sig2 = {
            'dtype': 'float64',
            'mean': 105.0,
            'std': 12.0,
            'unique_count': 48
        }
        
        # Compare signatures
        similarity = self.comparator._compare_signatures(sig1, sig2)
        
        # Should be reasonably similar
        self.assertGreater(similarity, 0.5)
        
        # Test different data types
        sig3 = {
            'dtype': 'object',
            'unique_count': 50
        }
        
        similarity_diff_type = self.comparator._compare_signatures(sig1, sig3)
        self.assertEqual(similarity_diff_type, 0.0)  # Different types should have 0 similarity
    
    @patch('comparisons.TableComparator._semantic_comparison')
    @patch('comparisons.TableComparator._fuzzy_comparison')
    @patch('comparisons.TableComparator._statistical_comparison')
    @patch('comparisons.TableComparator._content_comparison')
    def test_compare_tables(self, mock_content, mock_stat, mock_fuzzy, mock_semantic):
        """Test main table comparison method"""
        # Mock method returns
        mock_semantic.return_value = {'method': 'semantic', 'matches': [], 'statistics': {}}
        mock_fuzzy.return_value = {'method': 'fuzzy', 'matches': [], 'statistics': {}}
        mock_stat.return_value = {'method': 'statistical', 'matches': [], 'statistics': {}}
        mock_content.return_value = {'method': 'content', 'matches': [], 'statistics': {}}
        
        # Test single method
        results = self.comparator.compare_tables(
            self.table1, 
            self.table2, 
            methods=['semantic']
        )
        
        # Assertions
        self.assertIn('table1_columns', results)
        self.assertIn('table2_columns', results)
        self.assertIn('comparisons', results)
        self.assertIn('semantic', results['comparisons'])
        mock_semantic.assert_called_once()
        
        # Test multiple methods
        results_multi = self.comparator.compare_tables(
            self.table1, 
            self.table2, 
            methods=['semantic', 'fuzzy', 'statistical']
        )
        
        self.assertEqual(len(results_multi['comparisons']), 3)

class TestFuzzyComparison(unittest.TestCase):
    """Test fuzzy string matching functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.comparator = TableComparator()
        
        # Simple test tables for fuzzy matching
        self.table1 = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'user_name': ['John', 'Jane', 'Bob'],
            'email_addr': ['john@test.com', 'jane@test.com', 'bob@test.com']
        })
        
        self.table2 = pd.DataFrame({
            'cust_id': [1, 2, 3],
            'username': ['Alice', 'Charlie', 'Diana'],
            'email_address': ['alice@demo.com', 'charlie@demo.com', 'diana@demo.com']
        })
    
    def test_fuzzy_comparison_basic(self):
        """Test basic fuzzy comparison functionality"""
        # This test will work even if fuzzywuzzy is not available
        # by using a simplified comparison
        try:
            result = self.comparator._fuzzy_comparison(self.table1, self.table2)
            
            # Basic structure checks
            self.assertEqual(result['method'], 'fuzzy')
            self.assertIn('similarity_matrix', result)
            self.assertIn('matches', result)
            self.assertIn('statistics', result)
            
            # Check matrix dimensions
            similarity_matrix = np.array(result['similarity_matrix'])
            self.assertEqual(similarity_matrix.shape, (3, 3))
            
        except ImportError:
            # Skip test if fuzzywuzzy not available
            self.skipTest("fuzzywuzzy not available")

class TestStatisticalComparison(unittest.TestCase):
    """Test statistical comparison functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.comparator = TableComparator()
        
        # Create tables with similar statistical properties
        np.random.seed(42)
        
        self.table1 = pd.DataFrame({
            'numeric_col1': np.random.normal(100, 10, 100),
            'numeric_col2': np.random.uniform(0, 1, 100),
            'text_col1': ['text'] * 100,
            'bool_col1': np.random.choice([True, False], 100)
        })
        
        self.table2 = pd.DataFrame({
            'num_col1': np.random.normal(105, 12, 100),  # Similar to numeric_col1
            'num_col2': np.random.uniform(0.1, 0.9, 100),  # Similar to numeric_col2
            'string_col1': ['string'] * 100,  # Similar to text_col1
            'boolean_col1': np.random.choice([True, False], 100)  # Similar to bool_col1
        })
    
    def test_statistical_comparison_basic(self):
        """Test basic statistical comparison"""
        result = self.comparator._statistical_comparison(self.table1, self.table2)
        
        # Basic structure checks
        self.assertEqual(result['method'], 'statistical')
        self.assertIn('similarity_matrix', result)
        self.assertIn('matches', result)
        self.assertIn('statistics', result)
        self.assertIn('signatures1', result)
        self.assertIn('signatures2', result)
        
        # Check signatures
        sigs1 = result['signatures1']
        sigs2 = result['signatures2']
        
        self.assertEqual(len(sigs1), 4)
        self.assertEqual(len(sigs2), 4)

if __name__ == '__main__':
    # Set up test environment
    import warnings
    warnings.filterwarnings('ignore')
    
    # Run tests
    unittest.main(verbosity=2)