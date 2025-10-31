"""
Table Comparison Module
Author: Table Comparison Project
Date: October 31, 2025
Description: Handles various table comparison methods and ensemble approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import re

from config import *
from .embeddings import EmbeddingGenerator

class TableComparator:
    """
    Main class for comparing tables using various methods
    Supports semantic, fuzzy, statistical, and ensemble approaches
    """
    
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize table comparator
        
        Args:
            embedding_model: Name of embedding model to use
        """
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.comparison_cache = {}
    
    def compare_tables(
        self, 
        table1: pd.DataFrame, 
        table2: pd.DataFrame,
        methods: List[str] = ['ensemble'],
        weight_config: str = 'balanced'
    ) -> Dict:
        """
        Compare two tables using specified methods
        
        Args:
            table1: First table to compare
            table2: Second table to compare
            methods: List of comparison methods to use
            weight_config: Weight configuration for ensemble method
            
        Returns:
            Dictionary containing comparison results
        """
        results = {
            'table1_columns': list(table1.columns),
            'table2_columns': list(table2.columns),
            'table1_shape': table1.shape,
            'table2_shape': table2.shape,
            'methods_used': methods,
            'comparisons': {}
        }
        
        for method in methods:
            if method == 'semantic':
                results['comparisons']['semantic'] = self._semantic_comparison(table1, table2)
            elif method == 'fuzzy':
                results['comparisons']['fuzzy'] = self._fuzzy_comparison(table1, table2)
            elif method == 'statistical':
                results['comparisons']['statistical'] = self._statistical_comparison(table1, table2)
            elif method == 'content':
                results['comparisons']['content'] = self._content_comparison(table1, table2)
            elif method == 'hybrid':
                results['comparisons']['hybrid'] = self._hybrid_comparison(table1, table2)
            elif method == 'ensemble':
                results['comparisons']['ensemble'] = self._ensemble_comparison(
                    table1, table2, weight_config
                )
        
        return results
    
    def _semantic_comparison(self, table1: pd.DataFrame, table2: pd.DataFrame) -> Dict:
        """Perform semantic comparison using embeddings"""
        print("Computing semantic embeddings...")
        
        # Get embeddings for column names
        emb1 = self.embedding_generator.get_column_name_embeddings(list(table1.columns))
        emb2 = self.embedding_generator.get_column_name_embeddings(list(table2.columns))
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(emb1, emb2)
        
        # Find matches above threshold
        matches = self._extract_matches(
            similarity_matrix, 
            list(table1.columns), 
            list(table2.columns),
            SIMILARITY_THRESHOLDS['high_confidence']
        )
        
        return {
            'method': 'semantic',
            'similarity_matrix': similarity_matrix.tolist(),
            'matches': matches,
            'statistics': self._compute_match_statistics(matches)
        }
    
    def _fuzzy_comparison(self, table1: pd.DataFrame, table2: pd.DataFrame) -> Dict:
        """Perform fuzzy string matching comparison"""
        print("Computing fuzzy string similarities...")
        
        cols1 = list(table1.columns)
        cols2 = list(table2.columns)
        
        # Create similarity matrix using fuzzy matching
        similarity_matrix = np.zeros((len(cols1), len(cols2)))
        
        for i, col1 in enumerate(cols1):
            for j, col2 in enumerate(cols2):
                # Use ratio for overall similarity
                similarity = fuzz.ratio(col1.lower(), col2.lower()) / 100.0
                similarity_matrix[i, j] = similarity
        
        # Find matches above threshold
        matches = self._extract_matches(
            similarity_matrix,
            cols1,
            cols2,
            SIMILARITY_THRESHOLDS['medium_confidence']
        )
        
        return {
            'method': 'fuzzy',
            'similarity_matrix': similarity_matrix.tolist(),
            'matches': matches,
            'statistics': self._compute_match_statistics(matches)
        }
    
    def _statistical_comparison(self, table1: pd.DataFrame, table2: pd.DataFrame) -> Dict:
        """Perform statistical signature comparison"""
        print("Computing statistical signatures...")
        
        # Generate statistical signatures for each column
        sig1 = self._generate_statistical_signatures(table1)
        sig2 = self._generate_statistical_signatures(table2)
        
        # Compute similarity matrix based on statistical signatures
        similarity_matrix = self._compute_statistical_similarity(sig1, sig2)
        
        # Find matches above threshold
        matches = self._extract_matches(
            similarity_matrix,
            list(table1.columns),
            list(table2.columns),
            SIMILARITY_THRESHOLDS['statistical_threshold']
        )
        
        return {
            'method': 'statistical',
            'similarity_matrix': similarity_matrix.tolist(),
            'matches': matches,
            'statistics': self._compute_match_statistics(matches),
            'signatures1': sig1,
            'signatures2': sig2
        }
    
    def _content_comparison(self, table1: pd.DataFrame, table2: pd.DataFrame) -> Dict:
        """Perform content-based comparison using data embeddings"""
        print("Computing content-based embeddings...")
        
        # Get content embeddings
        emb1 = self.embedding_generator.get_content_embeddings(table1)
        emb2 = self.embedding_generator.get_content_embeddings(table2)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(emb1, emb2)
        
        # Find matches above threshold
        matches = self._extract_matches(
            similarity_matrix,
            list(table1.columns),
            list(table2.columns),
            SIMILARITY_THRESHOLDS['content_threshold']
        )
        
        return {
            'method': 'content',
            'similarity_matrix': similarity_matrix.tolist(),
            'matches': matches,
            'statistics': self._compute_match_statistics(matches)
        }
    
    def _hybrid_comparison(self, table1: pd.DataFrame, table2: pd.DataFrame) -> Dict:
        """Perform hybrid comparison combining name and content embeddings"""
        print("Computing hybrid embeddings...")
        
        # Get hybrid embeddings
        emb1 = self.embedding_generator.get_hybrid_embeddings(list(table1.columns), table1)
        emb2 = self.embedding_generator.get_hybrid_embeddings(list(table2.columns), table2)
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(emb1, emb2)
        
        # Find matches above threshold
        matches = self._extract_matches(
            similarity_matrix,
            list(table1.columns),
            list(table2.columns),
            SIMILARITY_THRESHOLDS['high_confidence']
        )
        
        return {
            'method': 'hybrid',
            'similarity_matrix': similarity_matrix.tolist(),
            'matches': matches,
            'statistics': self._compute_match_statistics(matches)
        }
    
    def _ensemble_comparison(
        self, 
        table1: pd.DataFrame, 
        table2: pd.DataFrame, 
        weight_config: str
    ) -> Dict:
        """Perform ensemble comparison combining multiple methods"""
        print("Computing ensemble comparison...")
        
        # Get individual method results
        semantic_result = self._semantic_comparison(table1, table2)
        fuzzy_result = self._fuzzy_comparison(table1, table2)
        statistical_result = self._statistical_comparison(table1, table2)
        content_result = self._content_comparison(table1, table2)
        
        # Get weights
        weights = get_weight_config(weight_config)
        
        # Combine similarity matrices
        combined_matrix = (
            weights['semantic'] * np.array(semantic_result['similarity_matrix']) +
            weights['fuzzy'] * np.array(fuzzy_result['similarity_matrix']) +
            weights['statistical'] * np.array(statistical_result['similarity_matrix']) +
            weights['content'] * np.array(content_result['similarity_matrix'])
        )
        
        # Find matches above threshold
        matches = self._extract_matches(
            combined_matrix,
            list(table1.columns),
            list(table2.columns),
            SIMILARITY_THRESHOLDS['high_confidence']
        )
        
        return {
            'method': 'ensemble',
            'weight_config': weight_config,
            'weights_used': weights,
            'similarity_matrix': combined_matrix.tolist(),
            'matches': matches,
            'statistics': self._compute_match_statistics(matches),
            'component_results': {
                'semantic': semantic_result,
                'fuzzy': fuzzy_result,
                'statistical': statistical_result,
                'content': content_result
            }
        }
    
    def _compute_similarity_matrix(self, emb1: Dict, emb2: Dict) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of embeddings"""
        cols1 = list(emb1.keys())
        cols2 = list(emb2.keys())
        
        # Create embedding matrices
        matrix1 = np.array([emb1[col] for col in cols1])
        matrix2 = np.array([emb2[col] for col in cols2])
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(matrix1, matrix2)
        
        return similarity_matrix
    
    def _extract_matches(
        self, 
        similarity_matrix: np.ndarray, 
        cols1: List[str], 
        cols2: List[str],
        threshold: float
    ) -> List[Dict]:
        """Extract matches from similarity matrix above threshold"""
        matches = []
        
        for i, col1 in enumerate(cols1):
            for j, col2 in enumerate(cols2):
                similarity = similarity_matrix[i, j]
                
                if similarity >= threshold:
                    matches.append({
                        'table1_column': col1,
                        'table2_column': col2,
                        'similarity': float(similarity),
                        'confidence': self._classify_confidence(similarity)
                    })
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches
    
    def _compute_match_statistics(self, matches: List[Dict]) -> Dict:
        """Compute statistics for a set of matches"""
        if not matches:
            return {
                'total_matches': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'confidence_distribution': {}
            }
        
        similarities = [match['similarity'] for match in matches]
        confidences = [match['confidence'] for match in matches]
        
        # Count confidence levels
        confidence_dist = {}
        for conf in confidences:
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        return {
            'total_matches': len(matches),
            'avg_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'confidence_distribution': confidence_dist
        }
    
    def _classify_confidence(self, similarity: float) -> str:
        """Classify similarity score into confidence levels"""
        if similarity >= SIMILARITY_THRESHOLDS['very_high_confidence']:
            return 'very_high'
        elif similarity >= SIMILARITY_THRESHOLDS['high_confidence']:
            return 'high'
        elif similarity >= SIMILARITY_THRESHOLDS['medium_confidence']:
            return 'medium'
        elif similarity >= SIMILARITY_THRESHOLDS['low_confidence']:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_statistical_signatures(self, df: pd.DataFrame) -> Dict:
        """Generate statistical signatures for each column"""
        signatures = {}
        
        for col in df.columns:
            series = df[col]
            sig = {}
            
            # Basic statistics
            sig['non_null_count'] = series.count()
            sig['null_count'] = series.isnull().sum()
            sig['unique_count'] = series.nunique()
            sig['dtype'] = str(series.dtype)
            
            # Type-specific statistics
            if np.issubdtype(series.dtype, np.number):
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    sig.update({
                        'mean': non_null_series.mean(),
                        'std': non_null_series.std(),
                        'min': non_null_series.min(),
                        'max': non_null_series.max(),
                        'median': non_null_series.median(),
                        'skewness': stats.skew(non_null_series),
                        'kurtosis': stats.kurtosis(non_null_series)
                    })
            elif series.dtype == 'object':
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    str_lengths = non_null_series.astype(str).str.len()
                    sig.update({
                        'avg_length': str_lengths.mean(),
                        'std_length': str_lengths.std(),
                        'min_length': str_lengths.min(),
                        'max_length': str_lengths.max()
                    })
            
            signatures[col] = sig
        
        return signatures
    
    def _compute_statistical_similarity(self, sig1: Dict, sig2: Dict) -> np.ndarray:
        """Compute similarity matrix based on statistical signatures"""
        cols1 = list(sig1.keys())
        cols2 = list(sig2.keys())
        
        similarity_matrix = np.zeros((len(cols1), len(cols2)))
        
        for i, col1 in enumerate(cols1):
            for j, col2 in enumerate(cols2):
                similarity = self._compare_signatures(sig1[col1], sig2[col2])
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _compare_signatures(self, sig1: Dict, sig2: Dict) -> float:
        """Compare two statistical signatures"""
        # Start with dtype compatibility
        if sig1['dtype'] != sig2['dtype']:
            # Check if both are numeric
            numeric_types = ['int64', 'int32', 'float64', 'float32']
            if not (sig1['dtype'] in numeric_types and sig2['dtype'] in numeric_types):
                return 0.0
        
        similarity_scores = []
        
        # Compare basic statistics
        for key in ['non_null_count', 'null_count', 'unique_count']:
            if key in sig1 and key in sig2:
                val1, val2 = sig1[key], sig2[key]
                if val1 + val2 > 0:  # Avoid division by zero
                    similarity = 1 - abs(val1 - val2) / max(val1 + val2, 1)
                    similarity_scores.append(similarity)
        
        # Compare numeric statistics if available
        numeric_keys = ['mean', 'std', 'min', 'max', 'median']
        for key in numeric_keys:
            if key in sig1 and key in sig2:
                val1, val2 = sig1[key], sig2[key]
                if pd.notna(val1) and pd.notna(val2):
                    # Normalize by range for comparison
                    range_val = max(abs(val1), abs(val2), 1)
                    similarity = 1 - abs(val1 - val2) / range_val
                    similarity_scores.append(max(0, similarity))
        
        # Compare string statistics if available
        string_keys = ['avg_length', 'std_length', 'min_length', 'max_length']
        for key in string_keys:
            if key in sig1 and key in sig2:
                val1, val2 = sig1[key], sig2[key]
                if pd.notna(val1) and pd.notna(val2) and (val1 + val2) > 0:
                    similarity = 1 - abs(val1 - val2) / (val1 + val2)
                    similarity_scores.append(max(0, similarity))
        
        # Return average similarity
        return np.mean(similarity_scores) if similarity_scores else 0.0