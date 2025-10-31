"""
Embedding Generation Module
Author: Table Comparison Project
Date: October 31, 2025
Description: Handles generation of embeddings using Ollama and other models
"""

import numpy as np
import pandas as pd
import requests
import json
import re
from typing import List, Dict, Optional, Union, Tuple
from config import *

class EmbeddingGenerator:
    """
    Handles embedding generation for table comparison
    Supports multiple embedding models and strategies
    """
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.cache = {} if CACHE_EMBEDDINGS else None
        
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a single text string
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array or None if failed
        """
        # Check cache first
        if self.cache is not None and text in self.cache:
            return self.cache[text]
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=TIMEOUT_SECONDS
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json()['embedding'])
                
                # Cache the result
                if self.cache is not None:
                    self.cache[text] = embedding
                
                return embedding
            else:
                print(f"Error getting embedding: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
            return None
    
    def get_column_name_embeddings(self, columns: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for column names
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to embeddings
        """
        embeddings = {}
        
        for col in columns:
            # Clean and enhance column name
            enhanced_name = self._enhance_column_name(col)
            embedding = self.get_embedding(enhanced_name)
            
            if embedding is not None:
                embeddings[col] = embedding
            else:
                print(f"Failed to get embedding for column: {col}")
        
        return embeddings
    
    def get_content_embeddings(self, df: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> Dict[str, np.ndarray]:
        """
        Generate embeddings based on column content
        
        Args:
            df: DataFrame to analyze
            sample_size: Number of samples to use for content analysis
            
        Returns:
            Dictionary mapping column names to content-based embeddings
        """
        content_embeddings = {}
        
        for col in df.columns:
            # Generate rich content description
            content_desc = self._generate_content_description(df[col], sample_size)
            embedding = self.get_embedding(content_desc)
            
            if embedding is not None:
                content_embeddings[col] = embedding
            else:
                print(f"Failed to get content embedding for column: {col}")
        
        return content_embeddings
    
    def get_hybrid_embeddings(
        self, 
        columns: List[str], 
        df: pd.DataFrame, 
        name_weight: float = HYBRID_WEIGHTS['name_weight'],
        content_weight: float = HYBRID_WEIGHTS['content_weight']
    ) -> Dict[str, np.ndarray]:
        """
        Generate hybrid embeddings combining name and content
        
        Args:
            columns: List of column names
            df: DataFrame for content analysis
            name_weight: Weight for name-based embeddings
            content_weight: Weight for content-based embeddings
            
        Returns:
            Dictionary mapping column names to hybrid embeddings
        """
        name_embeddings = self.get_column_name_embeddings(columns)
        content_embeddings = self.get_content_embeddings(df)
        
        hybrid_embeddings = {}
        
        for col in columns:
            if col in name_embeddings and col in content_embeddings:
                # Weighted combination
                hybrid_emb = (name_weight * name_embeddings[col] + 
                             content_weight * content_embeddings[col])
                hybrid_embeddings[col] = hybrid_emb
            elif col in name_embeddings:
                hybrid_embeddings[col] = name_embeddings[col]
            elif col in content_embeddings:
                hybrid_embeddings[col] = content_embeddings[col]
        
        return hybrid_embeddings
    
    def _enhance_column_name(self, column_name: str) -> str:
        """
        Enhance column name with context for better embeddings
        
        Args:
            column_name: Original column name
            
        Returns:
            Enhanced description
        """
        # Convert camelCase and snake_case to readable format
        readable_name = re.sub(r'([A-Z])', r' \1', column_name)
        readable_name = readable_name.replace('_', ' ').strip()
        
        # Create context-rich description
        enhanced = f"Database column representing {readable_name}. "
        
        # Add semantic hints based on common patterns
        name_lower = column_name.lower()
        
        if any(word in name_lower for word in ['id', 'key', 'pk']):
            enhanced += "Unique identifier field. "
        elif any(word in name_lower for word in ['name', 'title', 'label']):
            enhanced += "Text label or name field. "
        elif any(word in name_lower for word in ['date', 'time', 'created', 'updated']):
            enhanced += "Temporal/datetime field. "
        elif any(word in name_lower for word in ['price', 'cost', 'amount', 'value']):
            enhanced += "Monetary or numeric value field. "
        elif any(word in name_lower for word in ['email', 'mail']):
            enhanced += "Email address field. "
        elif any(word in name_lower for word in ['phone', 'tel', 'mobile']):
            enhanced += "Phone number field. "
        elif any(word in name_lower for word in ['address', 'location', 'city', 'state']):
            enhanced += "Geographic/address field. "
        elif any(word in name_lower for word in ['count', 'num', 'quantity']):
            enhanced += "Numeric count or quantity field. "
        elif any(word in name_lower for word in ['status', 'state', 'flag']):
            enhanced += "Status or categorical field. "
        
        enhanced += f"Field name: {column_name}"
        
        return enhanced
    
    def _generate_content_description(self, series: pd.Series, sample_size: int) -> str:
        """
        Generate rich description of column content
        
        Args:
            series: Pandas series to analyze
            sample_size: Number of samples to include
            
        Returns:
            Rich content description
        """
        # Basic statistics
        total_count = len(series)
        non_null_count = series.count()
        null_count = total_count - non_null_count
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        
        # Data type
        dtype_str = str(series.dtype)
        
        # Description start
        description = f"Column contains {dtype_str} data with {non_null_count} non-null values out of {total_count} total ({null_percentage:.1f}% null). "
        
        # Type-specific analysis
        if series.dtype in ['object', 'string']:
            description += self._analyze_text_column(series, sample_size)
        elif np.issubdtype(series.dtype, np.number):
            description += self._analyze_numeric_column(series)
        elif np.issubdtype(series.dtype, np.datetime64):
            description += self._analyze_datetime_column(series)
        elif series.dtype == 'bool':
            description += self._analyze_boolean_column(series)
        else:
            description += self._analyze_categorical_column(series, sample_size)
        
        return description
    
    def _analyze_text_column(self, series: pd.Series, sample_size: int) -> str:
        """Analyze text column content"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return "All values are null."
        
        # Sample values
        sample_values = non_null_series.sample(min(sample_size, len(non_null_series)), random_state=RANDOM_SEED)
        samples_str = ', '.join([f'"{str(val)[:MAX_SAMPLE_LENGTH]}"' for val in sample_values[:10]])
        
        # Length statistics
        lengths = non_null_series.astype(str).str.len()
        avg_length = lengths.mean()
        
        # Unique values
        unique_count = non_null_series.nunique()
        unique_percentage = (unique_count / len(non_null_series)) * 100
        
        # Pattern detection
        patterns = self._detect_patterns(non_null_series)
        pattern_desc = ", ".join(patterns) if patterns else "mixed text"
        
        description = (
            f"Text column with average length {avg_length:.1f} characters. "
            f"{unique_count} unique values ({unique_percentage:.1f}% unique). "
            f"Content patterns: {pattern_desc}. "
            f"Sample values: {samples_str}."
        )
        
        return description
    
    def _analyze_numeric_column(self, series: pd.Series) -> str:
        """Analyze numeric column content"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return "All values are null."
        
        # Basic statistics
        stats = {
            'min': non_null_series.min(),
            'max': non_null_series.max(),
            'mean': non_null_series.mean(),
            'median': non_null_series.median(),
            'std': non_null_series.std()
        }
        
        # Check for patterns
        is_integer = all(float(x).is_integer() for x in non_null_series if pd.notna(x))
        is_positive = all(x >= 0 for x in non_null_series if pd.notna(x))
        is_binary = set(non_null_series.unique()).issubset({0, 1})
        
        description = (
            f"Numeric column ranging from {stats['min']:.2f} to {stats['max']:.2f}, "
            f"mean {stats['mean']:.2f}, median {stats['median']:.2f}. "
        )
        
        if is_binary:
            description += "Binary values (0/1). "
        elif is_integer:
            description += "Integer values. "
        else:
            description += "Decimal values. "
        
        if is_positive:
            description += "All positive values. "
        
        return description
    
    def _analyze_datetime_column(self, series: pd.Series) -> str:
        """Analyze datetime column content"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return "All values are null."
        
        min_date = non_null_series.min()
        max_date = non_null_series.max()
        date_range = max_date - min_date
        
        description = (
            f"Datetime column spanning from {min_date} to {max_date} "
            f"({date_range.days} days range). "
        )
        
        return description
    
    def _analyze_boolean_column(self, series: pd.Series) -> str:
        """Analyze boolean column content"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return "All values are null."
        
        true_count = non_null_series.sum()
        false_count = len(non_null_series) - true_count
        true_percentage = (true_count / len(non_null_series)) * 100
        
        description = (
            f"Boolean column with {true_count} True values ({true_percentage:.1f}%) "
            f"and {false_count} False values. "
        )
        
        return description
    
    def _analyze_categorical_column(self, series: pd.Series, sample_size: int) -> str:
        """Analyze categorical column content"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return "All values are null."
        
        unique_values = non_null_series.unique()
        unique_count = len(unique_values)
        
        # Sample values
        sample_values = list(unique_values[:min(sample_size, len(unique_values))])
        samples_str = ', '.join([f'"{str(val)}"' for val in sample_values[:10]])
        
        description = (
            f"Categorical column with {unique_count} unique categories. "
            f"Categories include: {samples_str}. "
        )
        
        return description
    
    def _detect_patterns(self, series: pd.Series) -> List[str]:
        """Detect common patterns in text data"""
        patterns_found = []
        sample_size = min(100, len(series))
        sample = series.sample(sample_size, random_state=RANDOM_SEED)
        
        for pattern_name, pattern_regex in DETECTION_PATTERNS.items():
            matches = sum(1 for val in sample if re.match(pattern_regex, str(val)))
            if matches > sample_size * 0.5:  # More than 50% match
                patterns_found.append(pattern_name)
        
        return patterns_found