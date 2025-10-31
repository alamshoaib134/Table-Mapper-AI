"""
Configuration settings for table comparison project
Author: Table Comparison Project
Date: October 31, 2025
Description: Centralized configuration for all comparison methods and settings
"""

import os
from typing import Dict, List

# ============================================================================
# PROJECT METADATA
# ============================================================================
PROJECT_NAME = "Table Comparison with Advanced Embedding Techniques"
VERSION = "1.0.0"
AUTHOR = "Table Comparison Project"
DESCRIPTION = "Comprehensive Python toolkit for intelligent table comparison and column mapping"

# ============================================================================
# OLLAMA SETTINGS
# ============================================================================
DEFAULT_EMBEDDING_MODEL = 'mxbai-embed-large'
FALLBACK_MODELS = ['nomic-embed-text', 'bge-base-en-v1.5', 'all-minilm']

# Model specifications
EMBEDDING_MODELS = {
    'mxbai-embed-large': {
        'dimensions': 1024,
        'best_for': 'High accuracy, general purpose',
        'speed': 'medium',
        'size': '669MB'
    },
    'nomic-embed-text': {
        'dimensions': 768,
        'best_for': 'Fast, good for text similarity',
        'speed': 'fast',
        'size': '274MB'
    },
    'bge-base-en-v1.5': {
        'dimensions': 768,
        'best_for': 'Good balance of speed/accuracy',
        'speed': 'fast',
        'size': '68MB'
    },
    'all-minilm': {
        'dimensions': 384,
        'best_for': 'Fastest, good for simple tasks',
        'speed': 'very_fast',
        'size': '90MB'
    }
}

# ============================================================================
# DATA GENERATION SETTINGS
# ============================================================================
RANDOM_SEED = 42
DEFAULT_ROWS = 150
SAMPLE_SIZE = 50
MAX_SAMPLE_LENGTH = 50  # Maximum length of sample strings

# ============================================================================
# COMPARISON THRESHOLDS
# ============================================================================
SIMILARITY_THRESHOLDS = {
    'very_high_confidence': 0.8,
    'high_confidence': 0.7,
    'medium_confidence': 0.5,
    'low_confidence': 0.3,
    'statistical_threshold': 0.3,
    'content_threshold': 0.1
}

# ============================================================================
# ENSEMBLE WEIGHTS
# ============================================================================
ENSEMBLE_WEIGHTS = {
    'semantic': 0.35,      # Semantic embeddings weight
    'fuzzy': 0.25,         # Fuzzy string matching weight
    'statistical': 0.25,   # Statistical signatures weight
    'content': 0.15        # Content-based matching weight
}

# Alternative weight configurations
WEIGHT_CONFIGS = {
    'semantic_focused': {
        'semantic': 0.6,
        'fuzzy': 0.2,
        'statistical': 0.15,
        'content': 0.05
    },
    'balanced': {
        'semantic': 0.35,
        'fuzzy': 0.25,
        'statistical': 0.25,
        'content': 0.15
    },
    'content_focused': {
        'semantic': 0.2,
        'fuzzy': 0.15,
        'statistical': 0.25,
        'content': 0.4
    }
}

# ============================================================================
# HYBRID EMBEDDING SETTINGS
# ============================================================================
HYBRID_WEIGHTS = {
    'name_weight': 0.4,
    'content_weight': 0.6
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
FIGURE_SIZE = (20, 16)
LARGE_HEATMAP_SIZE = (30, 24)
DPI = 100
COLOR_PALETTE = ['skyblue', 'lightgreen', 'salmon', 'orange', 'lightcoral']

VISUALIZATION_CONFIG = {
    'heatmap': {
        'cmap': 'viridis',
        'fmt': '.3f',
        'figsize': LARGE_HEATMAP_SIZE
    },
    'comparison_charts': {
        'figsize': FIGURE_SIZE,
        'alpha': 0.8,
        'grid_alpha': 0.3
    },
    'pie_charts': {
        'colors': ['lightblue', 'orange', 'lightcoral'],
        'startangle': 90
    }
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
BATCH_SIZE = 10  # For processing large datasets
MAX_WORKERS = 4  # For parallel processing
CACHE_EMBEDDINGS = True

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
OUTPUT_DIRS = {
    'results': 'results',
    'logs': 'logs',
    'figures': 'figures',
    'exports': 'exports'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'table_comparison.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5
}

# ============================================================================
# DATA TYPE MAPPING
# ============================================================================
DATA_TYPE_CATEGORIES = {
    'boolean': ['bool'],
    'categorical': ['object', 'string', 'category'],
    'numerical': ['int64', 'int32', 'float64', 'float32'],
    'datetime': ['datetime64[ns]', 'datetime', 'date']
}

# ============================================================================
# PATTERN DETECTION REGEX
# ============================================================================
DETECTION_PATTERNS = {
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'phone': r'[\+]?[1-9]?[0-9]{7,15}',
    'url': r'https?://[^\s<>"{}|\\^`[\]]+',
    'zip_code': r'^\d{5}(-\d{4})?$',
    'date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}',
    'numeric': r'^\d+(\.\d+)?$',
    'currency': r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$',
    'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
    'social_security': r'^\d{3}-\d{2}-\d{4}$',
    'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_model_config(model_name: str) -> Dict:
    """Get configuration for a specific model"""
    return EMBEDDING_MODELS.get(model_name, EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL])

def get_weight_config(config_name: str) -> Dict:
    """Get ensemble weight configuration"""
    return WEIGHT_CONFIGS.get(config_name, ENSEMBLE_WEIGHTS)

def create_output_dirs() -> None:
    """Create output directories if they don't exist"""
    for dir_name in OUTPUT_DIRS.values():
        os.makedirs(dir_name, exist_ok=True)

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config() -> bool:
    """Validate configuration settings"""
    # Check that ensemble weights sum to 1.0
    for config_name, weights in WEIGHT_CONFIGS.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"Warning: {config_name} weights sum to {total}, not 1.0")
            return False
    
    # Check that hybrid weights sum to 1.0
    hybrid_total = sum(HYBRID_WEIGHTS.values())
    if abs(hybrid_total - 1.0) > 0.01:
        print(f"Warning: Hybrid weights sum to {hybrid_total}, not 1.0")
        return False
    
    return True

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
    print("Configuration validation passed!")