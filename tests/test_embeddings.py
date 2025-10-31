"""
Unit tests for embedding generation functionality
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embeddings import EmbeddingGenerator
from config import DEFAULT_EMBEDDING_MODEL, RANDOM_SEED

class TestEmbeddingGenerator(unittest.TestCase):
    """Test cases for EmbeddingGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = EmbeddingGenerator()
        
        # Create sample DataFrame for testing
        self.sample_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'customer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'email_address': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
            'purchase_amount': [150.50, 200.75, 99.99, 300.00, 175.25],
            'order_date': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', '2023-01-19']),
            'is_premium': [True, False, True, True, False]
        })
    
    def test_initialization(self):
        """Test EmbeddingGenerator initialization"""
        # Test default initialization
        gen = EmbeddingGenerator()
        self.assertEqual(gen.model_name, DEFAULT_EMBEDDING_MODEL)
        
        # Test custom model initialization
        custom_model = 'nomic-embed-text'
        gen_custom = EmbeddingGenerator(custom_model)
        self.assertEqual(gen_custom.model_name, custom_model)
    
    @patch('embeddings.requests.post')
    def test_get_embedding_success(self, mock_post):
        """Test successful embedding generation"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'embedding': [0.1, 0.2, 0.3, 0.4]}
        mock_post.return_value = mock_response
        
        # Test embedding generation
        result = self.generator.get_embedding("test text")
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3, 0.4])
    
    @patch('embeddings.requests.post')
    def test_get_embedding_failure(self, mock_post):
        """Test embedding generation failure"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        # Test embedding generation
        result = self.generator.get_embedding("test text")
        
        # Assertions
        self.assertIsNone(result)
    
    def test_enhance_column_name(self):
        """Test column name enhancement"""
        # Test various column name patterns
        test_cases = [
            ('user_id', 'Database column representing user id. Unique identifier field. Field name: user_id'),
            ('customerName', 'Database column representing customer Name. Text label or name field. Field name: customerName'),
            ('emailAddress', 'Database column representing email Address. Email address field. Field name: emailAddress'),
            ('created_at', 'Database column representing created at. Temporal/datetime field. Field name: created_at'),
            ('total_price', 'Database column representing total price. Monetary or numeric value field. Field name: total_price')
        ]
        
        for column_name, expected_keywords in test_cases:
            enhanced = self.generator._enhance_column_name(column_name)
            self.assertIn('Database column representing', enhanced)
            self.assertIn(f'Field name: {column_name}', enhanced)
    
    def test_analyze_numeric_column(self):
        """Test numeric column analysis"""
        numeric_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        description = self.generator._analyze_numeric_column(numeric_series)
        
        # Check that description contains expected elements
        self.assertIn('Numeric column', description)
        self.assertIn('ranging from', description)
        self.assertIn('mean', description)
        self.assertIn('median', description)
    
    def test_analyze_text_column(self):
        """Test text column analysis"""
        text_series = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry'])
        description = self.generator._analyze_text_column(text_series, sample_size=3)
        
        # Check that description contains expected elements
        self.assertIn('Text column', description)
        self.assertIn('average length', description)
        self.assertIn('unique values', description)
        self.assertIn('Sample values', description)
    
    def test_analyze_boolean_column(self):
        """Test boolean column analysis"""
        bool_series = pd.Series([True, False, True, True, False])
        description = self.generator._analyze_boolean_column(bool_series)
        
        # Check that description contains expected elements
        self.assertIn('Boolean column', description)
        self.assertIn('True values', description)
        self.assertIn('False values', description)
    
    def test_analyze_datetime_column(self):
        """Test datetime column analysis"""
        date_series = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        description = self.generator._analyze_datetime_column(date_series)
        
        # Check that description contains expected elements
        self.assertIn('Datetime column', description)
        self.assertIn('spanning from', description)
        self.assertIn('days range', description)
    
    def test_detect_patterns(self):
        """Test pattern detection in text data"""
        # Test email pattern detection
        email_series = pd.Series(['test@email.com', 'user@domain.org', 'admin@site.net'])
        patterns = self.generator._detect_patterns(email_series)
        self.assertIn('email', patterns)
        
        # Test numeric pattern detection
        numeric_series = pd.Series(['123', '456', '789', '012'])
        patterns = self.generator._detect_patterns(numeric_series)
        self.assertIn('numeric', patterns)
    
    @patch('embeddings.requests.post')
    def test_get_column_name_embeddings(self, mock_post):
        """Test column name embedding generation"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'embedding': [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        columns = ['user_id', 'name', 'email']
        embeddings = self.generator.get_column_name_embeddings(columns)
        
        # Assertions
        self.assertEqual(len(embeddings), 3)
        for col in columns:
            self.assertIn(col, embeddings)
            self.assertIsInstance(embeddings[col], np.ndarray)
    
    @patch('embeddings.requests.post')
    def test_get_content_embeddings(self, mock_post):
        """Test content-based embedding generation"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'embedding': [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        embeddings = self.generator.get_content_embeddings(self.sample_df, sample_size=3)
        
        # Assertions
        self.assertEqual(len(embeddings), len(self.sample_df.columns))
        for col in self.sample_df.columns:
            self.assertIn(col, embeddings)
            self.assertIsInstance(embeddings[col], np.ndarray)
    
    @patch('embeddings.requests.post')
    def test_get_hybrid_embeddings(self, mock_post):
        """Test hybrid embedding generation"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'embedding': [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response
        
        columns = list(self.sample_df.columns)
        embeddings = self.generator.get_hybrid_embeddings(columns, self.sample_df)
        
        # Assertions
        self.assertEqual(len(embeddings), len(columns))
        for col in columns:
            self.assertIn(col, embeddings)
            self.assertIsInstance(embeddings[col], np.ndarray)
    
    def test_content_description_generation(self):
        """Test content description generation for different column types"""
        # Test each column type in sample DataFrame
        for col in self.sample_df.columns:
            description = self.generator._generate_content_description(
                self.sample_df[col], sample_size=3
            )
            
            # Basic checks
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 10)  # Should be descriptive
            self.assertIn('Column contains', description)
    
    def test_caching_functionality(self):
        """Test embedding caching functionality"""
        # Create generator with caching enabled
        gen_with_cache = EmbeddingGenerator()
        gen_with_cache.cache = {}
        
        # Manually add to cache
        test_text = "test text"
        test_embedding = np.array([0.1, 0.2, 0.3])
        gen_with_cache.cache[test_text] = test_embedding
        
        # Should return cached result without making request
        with patch('embeddings.requests.post') as mock_post:
            result = gen_with_cache.get_embedding(test_text)
            mock_post.assert_not_called()
            np.testing.assert_array_equal(result, test_embedding)

class TestEmbeddingIntegration(unittest.TestCase):
    """Integration tests for embedding functionality"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.generator = EmbeddingGenerator()
        
        # Create more complex test data
        self.test_df1 = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'full_name': ['John Smith', 'Jane Doe', 'Bob Wilson'],
            'contact_email': ['john@test.com', 'jane@test.com', 'bob@test.com'],
            'order_total': [100.0, 150.5, 200.0]
        })
        
        self.test_df2 = pd.DataFrame({
            'user_id': [1, 2, 3],
            'customer_name': ['Alice Brown', 'Charlie Green', 'Diana White'],
            'email_address': ['alice@demo.com', 'charlie@demo.com', 'diana@demo.com'],
            'purchase_amount': [120.0, 180.0, 90.0]
        })
    
    @patch('embeddings.requests.post')
    def test_end_to_end_comparison_setup(self, mock_post):
        """Test end-to-end embedding generation for comparison"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'embedding': list(np.random.rand(10))}
        mock_post.return_value = mock_response
        
        # Generate embeddings for both tables
        emb1 = self.generator.get_column_name_embeddings(list(self.test_df1.columns))
        emb2 = self.generator.get_column_name_embeddings(list(self.test_df2.columns))
        
        # Basic validation
        self.assertEqual(len(emb1), len(self.test_df1.columns))
        self.assertEqual(len(emb2), len(self.test_df2.columns))
        
        # All embeddings should be present and valid
        for col in self.test_df1.columns:
            self.assertIn(col, emb1)
            self.assertIsInstance(emb1[col], np.ndarray)
        
        for col in self.test_df2.columns:
            self.assertIn(col, emb2)
            self.assertIsInstance(emb2[col], np.ndarray)

if __name__ == '__main__':
    # Set up test environment
    import warnings
    warnings.filterwarnings('ignore')
    
    # Run tests
    unittest.main(verbosity=2)