# core/knowledge_weighting.py - FIXED for langchain_community.vectorstores
# Production Knowledge Grounding System with Dynamic Calibration

from typing import List, Dict, Tuple, Optional, Any
# from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_pinecone import Pinecone as PineconeVectorStore
from .llm_config import embeddings, INDEX_NAME
import numpy as np
from collections import defaultdict
import os
import json
from datetime import datetime, timedelta
from django.core.cache import cache
from pinecone import Pinecone
import random
import re


# ==================== PINECONE HELPER ====================

def get_pinecone_vectorstore(index_name: str = None, namespace: str = None):
    """
    Get PineconeVectorStore for langchain_community version.
    
    Args:
        index_name: Name of the Pinecone index
        namespace: Optional namespace
    
    Returns:
        PineconeVectorStore instance
    """
    if index_name is None:
        index_name = INDEX_NAME
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    # Create vector store using from_existing_index
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
    return vector_store


# ==================== DYNAMIC CATEGORY SYSTEM ====================

class DynamicCategoryMapper:
    """
    Dynamic, learning-based category normalization system.
    
    Features:
    - Auto-discovers categories from KB
    - No hardcoded category patterns
    - Uses fuzzy matching and semantic similarity
    - Learns from actual KB distribution
    - Supports unlimited categories
    """
    
    CACHE_KEY_CATEGORIES = "kb_dynamic_categories_v1"
    CACHE_KEY_PATTERNS = "kb_category_patterns_v1"
    CACHE_TIMEOUT = 7200  # 2 hours
    
    # Minimal seed patterns for bootstrapping (expandable)
    SEED_PATTERNS = {
        'mathematics': ['math', 'calculus', 'algebra'],
        'science': ['physics', 'chemistry', 'biology'],
        'programming': ['programming', 'coding', 'software'],
        'business': ['business', 'management', 'marketing'],
    }
    
    def __init__(self, enable_learning: bool = True, similarity_threshold: float = 0.9):
        """
        Initialize dynamic mapper.
        
        Args:
            enable_learning: Enable learning new patterns from KB
            similarity_threshold: Fuzzy matching threshold (0-1)
        """
        self.enable_learning = enable_learning
        self.similarity_threshold = similarity_threshold
        self._category_cache = None
        self._pattern_cache = None
    
    def normalize_category(self, category: str, learn: bool = True) -> str:
        """
        Normalize category name dynamically.
        
        Args:
            category: Input category string
            learn: Whether to learn from this input
        
        Returns:
            Normalized category name
        """
        if not category or not isinstance(category, str):
            return 'general'
        
        cat_clean = self._clean_category_string(category)
        
        if not cat_clean:
            return 'general'
        
        # Get current category mappings
        patterns = self._get_category_patterns()
        
        # Try exact match first
        if cat_clean in patterns:
            return cat_clean
        
        # Try fuzzy matching
        matched = self._fuzzy_match_category(cat_clean, patterns)
        
        if matched:
            # Learn this mapping for future use
            if learn and self.enable_learning:
                self._learn_category_mapping(cat_clean, matched)
            return matched
        
        # No match - treat as new category
        if learn and self.enable_learning:
            self._register_new_category(cat_clean)
        
        return cat_clean
    
    def _clean_category_string(self, category: str) -> str:
        """Clean and normalize category string."""
        # Convert to lowercase
        clean = category.lower().strip()
        
        # Remove special characters except spaces and underscores
        clean = ''.join(c if c.isalnum() or c in ' _-' else ' ' for c in clean)
        
        # Normalize whitespace
        clean = ' '.join(clean.split())
        
        # Replace spaces with underscores for consistency
        clean = clean.replace(' ', '_')
        
        # Remove leading/trailing underscores
        clean = clean.strip('_')
        
        return clean
    
    def _fuzzy_match_category(self, category: str, patterns: Dict) -> Optional[str]:
        """
        Fuzzy match category against known patterns.
        
        Args:
            category: Input category
            patterns: Current category patterns
        
        Returns:
            Matched category or None
        """
        # Tokenize input
        cat_tokens = set(category.replace('_', ' ').split())
        
        best_match = None
        best_score = 0.0
        
        for standard_cat, keywords in patterns.items():
            # Calculate overlap score
            keyword_tokens = set()
            for kw in keywords:
                keyword_tokens.update(kw.replace('_', ' ').split())
            
            # Intersection over union (Jaccard similarity)
            intersection = cat_tokens & keyword_tokens
            union = cat_tokens | keyword_tokens
            
            if union:
                score = len(intersection) / len(union)
                
                # Boost score if substring match exists
                if any(kw in category for kw in keywords):
                    score += 0.3
                
                # Boost score if exact token match
                if any(token in keywords for token in cat_tokens):
                    score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_match = standard_cat
        
        # Return match if above threshold
        if best_score >= self.similarity_threshold:
            return best_match
        
        return None
    
    def _get_category_patterns(self) -> Dict[str, List[str]]:
        """
        Get current category patterns (learned + seed).
        
        Returns:
            Dict mapping category -> list of keywords
        """
        # Try cache first
        if self._pattern_cache:
            return self._pattern_cache
        
        cached = cache.get(self.CACHE_KEY_PATTERNS)
        if cached:
            self._pattern_cache = cached
            return cached
        
        # Initialize with seed patterns
        patterns = dict(self.SEED_PATTERNS)
        
        # Load learned patterns from KB
        if self.enable_learning:
            learned = self._load_learned_patterns()
            
            # Merge learned patterns
            for cat, keywords in learned.items():
                if cat in patterns:
                    patterns[cat] = list(set(patterns[cat] + keywords))
                else:
                    patterns[cat] = keywords
        
        # Cache and return
        cache.set(self.CACHE_KEY_PATTERNS, patterns, self.CACHE_TIMEOUT)
        self._pattern_cache = patterns
        
        return patterns
    
    def _load_learned_patterns(self) -> Dict[str, List[str]]:
        """
        Load learned patterns from KB sampling.
        Discovers actual categories and their variations in the KB.
        """
        try:
            # FIXED: Use helper function
            vector_store = get_pinecone_vectorstore(INDEX_NAME)
            
            # Sample vectors to discover categories
            learned_patterns = defaultdict(set)
            
            # Use diverse queries to sample KB
            sample_queries = [
                "study learning education",
                "analysis research data",
                "development software code",
                "business finance economics"
            ]
            
            for query in sample_queries:
                try:
                    results = vector_store.similarity_search(query=query, k=50)
                    
                    for doc in results:
                        meta = doc.metadata
                        
                        # Extract category variations
                        for field in ['category', 'subject', 'domain', 'field', 'type']:
                            if field in meta and meta[field]:
                                cat_value = str(meta[field]).lower().strip()
                                
                                if cat_value and len(cat_value) > 2:
                                    # Extract main category and add as keyword
                                    main_cat = self._extract_main_category(cat_value)
                                    
                                    if main_cat:
                                        # Add variations as keywords
                                        learned_patterns[main_cat].add(cat_value)
                                        learned_patterns[main_cat].update(cat_value.split('_'))
                                        learned_patterns[main_cat].update(cat_value.split())
                
                except Exception as e:
                    print(f"⚠️ Pattern learning query failed: {e}")
                    continue
            
            # Convert sets to lists and filter
            result = {}
            for cat, keywords in learned_patterns.items():
                # Filter out very short or common words
                filtered = [kw for kw in keywords if len(kw) > 2 and kw not in ['the', 'and', 'or', 'for']]
                if filtered:
                    result[cat] = filtered[:20]  # Limit to top 20 keywords
            
            return result
            
        except Exception as e:
            print(f"⚠️ Pattern learning failed: {e}")
            return {}
    
    def _extract_main_category(self, category_value: str) -> Optional[str]:
        """
        Extract main category from complex category string.
        
        Examples:
            "computer_science_101" -> "computer_science"
            "advanced mathematics" -> "mathematics"
            "intro to physics" -> "physics"
        """
        # Remove common prefixes
        prefixes = ['intro', 'introduction', 'advanced', 'basic', 'intermediate', 'to']
        tokens = category_value.replace('_', ' ').split()
        
        # Filter out prefixes and numbers
        main_tokens = [
            t for t in tokens 
            if t not in prefixes and not t.isdigit() and len(t) > 2
        ]
        
        if not main_tokens:
            return None
        
        # Return first meaningful token or combined
        if len(main_tokens) == 1:
            return main_tokens[0]
        elif len(main_tokens) == 2:
            return '_'.join(main_tokens)
        else:
            # Take first two most meaningful tokens
            return '_'.join(main_tokens[:2])
    
    def _learn_category_mapping(self, input_category: str, matched_category: str):
        """
        Learn that input_category maps to matched_category.
        Updates patterns for future use.
        """
        try:
            patterns = self._get_category_patterns()
            
            # Add input as keyword for matched category
            if matched_category in patterns:
                if input_category not in patterns[matched_category]:
                    patterns[matched_category].append(input_category)
                    
                    # Update cache
                    cache.set(self.CACHE_KEY_PATTERNS, patterns, self.CACHE_TIMEOUT)
                    self._pattern_cache = patterns
                    
                    print(f"   📚 Learned: '{input_category}' → '{matched_category}'")
        
        except Exception as e:
            print(f"⚠️ Failed to learn mapping: {e}")
    
    def _register_new_category(self, category: str):
        """
        Register a new category discovered in data.
        """
        try:
            patterns = self._get_category_patterns()
            
            if category not in patterns:
                # Create new pattern entry
                patterns[category] = [category]
                
                # Add tokens as keywords
                tokens = category.replace('_', ' ').split()
                patterns[category].extend([t for t in tokens if len(t) > 2])
                
                # Update cache
                cache.set(self.CACHE_KEY_PATTERNS, patterns, self.CACHE_TIMEOUT)
                self._pattern_cache = patterns
                
                print(f"   🆕 New category registered: '{category}'")
        
        except Exception as e:
            print(f"⚠️ Failed to register category: {e}")
    
    def get_all_categories(self) -> List[str]:
        """
        Get all known categories.
        
        Returns:
            List of category names
        """
        patterns = self._get_category_patterns()
        return sorted(patterns.keys())
    
    def get_category_info(self, category: str) -> Dict:
        """
        Get information about a category.
        
        Returns:
            Dict with keywords, variations, etc.
        """
        patterns = self._get_category_patterns()
        
        if category in patterns:
            return {
                'category': category,
                'keywords': patterns[category],
                'keyword_count': len(patterns[category]),
                'is_learned': category not in self.SEED_PATTERNS,
                'exists': True
            }
        
        return {
            'category': category,
            'keywords': [],
            'keyword_count': 0,
            'is_learned': False,
            'exists': False
        }
    
    def refresh_patterns(self):
        """Force refresh of category patterns from KB."""
        cache.delete(self.CACHE_KEY_PATTERNS)
        self._pattern_cache = None
        self._get_category_patterns()
        print("✅ Category patterns refreshed")
    
    def export_patterns(self) -> Dict:
        """Export current patterns for backup/analysis."""
        return self._get_category_patterns()
    
    def import_patterns(self, patterns: Dict):
        """Import custom patterns."""
        cache.set(self.CACHE_KEY_PATTERNS, patterns, self.CACHE_TIMEOUT)
        self._pattern_cache = patterns
        print(f"✅ Imported {len(patterns)} category patterns")


# ==================== SCHEMA HANDLER ====================

class SchemaHandler:
    """
    Flexible schema handler with dynamic category normalization.
    """
    
    # Field synonyms
    FIELD_SYNONYMS = {
        'category': ['category', 'subject', 'domain', 'field', 'topic_area', 'discipline', 'type'],
        'complexity': ['complexity', 'difficulty', 'level', 'complexity_score', 'difficulty_level', 'grade'],
        'topic': ['topic', 'title', 'subject', 'theme', 'chapter', 'name'],
        'source_type': ['source_type', 'document_type', 'type', 'doc_type', 'material_type', 'kind'],
        'file': ['file', 'filename', 'source', 'document', 'doc_name', 'path'],
        'pages': ['pages', 'page_count', 'num_pages', 'total_pages', 'length'],
    }
    
    TYPE_DEFAULTS = {
        'str': 'unknown',
        'int': 0,
        'float': 0.0,
        'bool': False,
    }
    
    # Initialize dynamic category mapper (class-level singleton)
    _category_mapper = None
    
    @classmethod
    def get_category_mapper(cls) -> DynamicCategoryMapper:
        """Get or create category mapper instance."""
        if cls._category_mapper is None:
            cls._category_mapper = DynamicCategoryMapper(
                enable_learning=True,
                similarity_threshold=0.6  # Adjustable
            )
        return cls._category_mapper
    
    @classmethod
    def extract_field(cls, metadata: Dict, field_name: str, expected_type: str = 'str', default: Any = None) -> Any:
        """
        Flexibly extract field with type safety and fallbacks.
        """
        if not metadata:
            return default if default is not None else cls.TYPE_DEFAULTS.get(expected_type, 'unknown')
        
        # Try all synonyms
        synonyms = cls.FIELD_SYNONYMS.get(field_name, [field_name])
        
        for synonym in synonyms:
            if synonym in metadata and metadata[synonym] is not None:
                value = metadata[synonym]
                return cls._normalize_by_type(value, expected_type, field_name)
        
        return default if default is not None else cls.TYPE_DEFAULTS.get(expected_type, 'unknown')
    
    @classmethod
    def _normalize_by_type(cls, value: Any, expected_type: str, field_name: str) -> Any:
        """Type-aware normalization with dynamic category handling."""
        
        if expected_type == 'str':
            if isinstance(value, str):
                normalized = value.strip().lower()
                # Use dynamic category normalization
                if field_name == 'category':
                    mapper = cls.get_category_mapper()
                    return mapper.normalize_category(normalized, learn=True)
                return normalized
            return str(value).lower()
        
        elif expected_type == 'int':
            try:
                val = int(float(value))
                if field_name == 'complexity':
                    return max(1, min(10, val))
                return max(0, val)
            except (ValueError, TypeError):
                return 5 if field_name == 'complexity' else 0
        
        elif expected_type == 'float':
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        elif expected_type == 'bool':
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', 'yes', '1', 'y']
            return bool(value)
        
        return value
    
    @classmethod
    def standardize_metadata(cls, metadata: Dict) -> Dict:
        """
        Convert any metadata schema to standardized format.
        Uses dynamic category normalization.
        """
        return {
            'category': cls.extract_field(metadata, 'category', 'str'),
            'complexity': cls.extract_field(metadata, 'complexity', 'int'),
            'topic': cls.extract_field(metadata, 'topic', 'str'),
            'source_type': cls.extract_field(metadata, 'source_type', 'str'),
            'file': cls.extract_field(metadata, 'file', 'str'),
            'pages': cls.extract_field(metadata, 'pages', 'int'),
            '_original': metadata  # Preserve original
        }
    
    @classmethod
    def get_all_discovered_categories(cls) -> List[str]:
        """Get all categories discovered in KB."""
        mapper = cls.get_category_mapper()
        return mapper.get_all_categories()
    
    @classmethod
    def get_category_statistics(cls) -> Dict:
        """Get statistics about category mappings."""
        mapper = cls.get_category_mapper()
        all_cats = mapper.get_all_categories()
        
        stats = {
            'total_categories': len(all_cats),
            'categories': []
        }
        
        for cat in all_cats:
            info = mapper.get_category_info(cat)
            stats['categories'].append(info)
        
        return stats
    
    @classmethod
    def refresh_category_system(cls):
        """Refresh category discovery system."""
        mapper = cls.get_category_mapper()
        mapper.refresh_patterns()


# ==================== DYNAMIC CALIBRATION ENGINE ====================

class DynamicCalibrationEngine:
    """
    Dynamic, data-driven calibration that adapts to actual KB distribution.
    No hardcoded domain rules - learns from KB statistics.
    """
    
    def __init__(self, cache_timeout: int = 3600):
        """
        Initialize with adaptive calibration.
        
        Args:
            cache_timeout: Cache duration in seconds
        """
        self.cache_timeout = cache_timeout
        self.cache_key_calibration = "kb_dynamic_calibration_v1"
        
    def get_domain_parameters(self, category: str, force_refresh: bool = False) -> Dict:
        """
        Get calibration parameters for a category dynamically.
        Parameters are computed from actual KB statistics.
        
        Args:
            category: Document category
            force_refresh: Force recomputation
        
        Returns:
            Calibration parameters {base, spread, boost, confidence}
        """
        # Try cache first
        if not force_refresh:
            cached = cache.get(self.cache_key_calibration)
            if cached and category in cached:
                return cached[category]
        
        # Compute calibration parameters
        all_params = self._compute_all_calibration_parameters()
        
        # Cache results
        cache.set(self.cache_key_calibration, all_params, self.cache_timeout)
        
        # Return category-specific params or default
        return all_params.get(category, all_params.get('_default', self._default_parameters()))
    
    def _compute_all_calibration_parameters(self) -> Dict[str, Dict]:
        """
        Compute calibration parameters for all categories dynamically.
        Uses percentile-based statistics from actual KB data.
        """
        try:
            # Get category statistics
            from .knowledge_weighting import KnowledgeStatisticsEngine
            stats_engine = KnowledgeStatisticsEngine()
            
            category_stats = stats_engine.get_category_statistics()
            similarity_stats = stats_engine.get_similarity_statistics()
            
            all_params = {}
            
            for category, cat_info in category_stats.items():
                # Extract statistics
                doc_count = cat_info.get('document_count', 0)
                avg_similarity = cat_info.get('avg_similarity', 0.65)
                std_similarity = cat_info.get('std_similarity', 0.15)
                p50_similarity = cat_info.get('p50_similarity', 0.67)
                
                # Compute calibration parameters
                # Base: Use median similarity as baseline
                base = p50_similarity
                
                # Spread: Use std or default
                spread = max(std_similarity, 0.10)  # Minimum spread
                
                # Boost: Based on document frequency (rare categories get boost)
                total_docs = sum(c.get('document_count', 0) for c in category_stats.values())
                frequency = doc_count / total_docs if total_docs > 0 else 0.1
                
                # Inverse frequency boost (rare = higher importance)
                if frequency < 0.05:
                    boost = 1.30  # Very rare
                elif frequency < 0.10:
                    boost = 1.20  # Rare
                elif frequency < 0.20:
                    boost = 1.10  # Less common
                else:
                    boost = 1.00  # Common
                
                # Confidence: Based on sample size
                confidence = min(doc_count / 100.0, 1.0)  # Max at 100+ docs
                
                all_params[category] = {
                    'base': base,
                    'spread': spread,
                    'boost': boost,
                    'confidence': confidence,
                    'doc_count': doc_count,
                    'frequency': frequency
                }
            
            # Add global default
            all_params['_default'] = self._compute_default_from_global(similarity_stats)
            
            return all_params
            
        except Exception as e:
            print(f"⚠️ Calibration computation failed: {e}")
            # Return safe defaults
            return {'_default': self._default_parameters()}
    
    def _compute_default_from_global(self, global_stats: Dict) -> Dict:
        """Compute default parameters from global statistics."""
        return {
            'base': global_stats.get('median', 0.65),
            'spread': global_stats.get('std', 0.15),
            'boost': 1.00,
            'confidence': 0.5,
            'doc_count': global_stats.get('total_samples', 0),
            'frequency': 1.0
        }
    
    def _default_parameters(self) -> Dict:
        """Fallback default parameters."""
        return {
            'base': 0.65,
            'spread': 0.20,
            'boost': 1.00,
            'confidence': 0.3,
            'doc_count': 0,
            'frequency': 0.0
        }
    
    def apply_calibration(self, score: float, category: str) -> float:
        """
        Apply dynamic calibration to a raw score.
        
        Args:
            score: Raw normalized score (0-1)
            category: Document category
        
        Returns:
            Calibrated score (0-1)
        """
        params = self.get_domain_parameters(category)
        
        # Z-score normalization
        z = (score - params['base']) / params['spread']
        
        # Sigmoid transformation
        calibrated = 1 / (1 + np.exp(-z))
        
        # Apply boost
        calibrated *= params['boost']
        
        return float(np.clip(calibrated, 0, 1))
    
    def clear_cache(self):
        """Clear calibration cache."""
        cache.delete(self.cache_key_calibration)


# ==================== ADAPTIVE THRESHOLDING ====================

class AdaptiveThresholdEngine:
    """
    Adaptive thresholding based on actual KB statistics.
    Thresholds automatically adjust to data distribution.
    """
    
    def __init__(self, thresholds_config: Optional[Dict] = None):
        """
        Initialize with optional custom thresholds.
        
        Args:
            thresholds_config: Override default thresholds
        """
        self.custom_thresholds = thresholds_config or {}
        self.cache_key = "kb_adaptive_thresholds_v1"
        self.cache_timeout = 3600
    
    def get_threshold(self, threshold_name: str, category: Optional[str] = None) -> float:
        """
        Get adaptive threshold value.
        
        Args:
            threshold_name: Name of threshold (e.g., 'min_similarity')
            category: Optional category for category-specific thresholds
        
        Returns:
            Threshold value
        """
        # Check custom override
        if threshold_name in self.custom_thresholds:
            return self.custom_thresholds[threshold_name]
        
        # Get adaptive thresholds from cache or compute
        thresholds = self._get_adaptive_thresholds()
        
        # Category-specific threshold
        if category and category in thresholds:
            return thresholds[category].get(threshold_name, thresholds['_global'].get(threshold_name, 0.5))
        
        # Global threshold
        return thresholds['_global'].get(threshold_name, 0.5)
    
    def _get_adaptive_thresholds(self) -> Dict:
        """Get or compute adaptive thresholds."""
        cached = cache.get(self.cache_key)
        if cached:
            return cached
        
        thresholds = self._compute_adaptive_thresholds()
        cache.set(self.cache_key, thresholds, self.cache_timeout)
        return thresholds
    
    def _compute_adaptive_thresholds(self) -> Dict:
        """
        Compute adaptive thresholds from KB statistics.
        Uses percentiles for robustness.
        """
        try:
            from .knowledge_weighting import KnowledgeStatisticsEngine
            stats_engine = KnowledgeStatisticsEngine()
            
            global_stats = stats_engine.get_similarity_statistics()
            
            # Compute global thresholds using percentiles
            global_thresholds = {
                'min_similarity': global_stats.get('p25', 0.30),  # 25th percentile
                'moderate_similarity': global_stats.get('p50', 0.50),  # Median
                'high_similarity': global_stats.get('p75', 0.75),  # 75th percentile
                'very_high_similarity': global_stats.get('p90', 0.85),  # 90th percentile
                
                'low_confidence': 0.30,  # Fixed for now
                'moderate_confidence': 0.50,
                'high_confidence': 0.70,
                
                'minimal_depth_score': global_stats.get('p25', 0.40),
                'moderate_depth_score': global_stats.get('p50', 0.60),
                'substantial_depth_score': global_stats.get('p75', 0.75),
                'extensive_depth_score': global_stats.get('p90', 0.85),
            }
            
            result = {'_global': global_thresholds}
            
            # Category-specific thresholds (optional)
            category_stats = stats_engine.get_category_statistics()
            for category, cat_info in category_stats.items():
                result[category] = {
                    'min_similarity': cat_info.get('p25_similarity', global_thresholds['min_similarity']),
                    'high_similarity': cat_info.get('p75_similarity', global_thresholds['high_similarity']),
                }
            
            return result
            
        except Exception as e:
            print(f"⚠️ Threshold computation failed: {e}")
            return {'_global': self._default_thresholds()}
    
    def _default_thresholds(self) -> Dict:
        """Fallback default thresholds."""
        return {
            'min_similarity': 0.30,
            'moderate_similarity': 0.50,
            'high_similarity': 0.70,
            'very_high_similarity': 0.85,
            'low_confidence': 0.30,
            'moderate_confidence': 0.50,
            'high_confidence': 0.70,
            'minimal_depth_score': 0.40,
            'moderate_depth_score': 0.60,
            'substantial_depth_score': 0.75,
            'extensive_depth_score': 0.85,
        }
    
    def assess_knowledge_depth(
        self, 
        avg_score: float, 
        num_docs: int, 
        category: Optional[str] = None
    ) -> str:
        """
        Assess knowledge depth using adaptive thresholds.
        
        Returns: 'extensive', 'substantial', 'moderate', 'limited', 'minimal', or 'none'
        """
        if num_docs == 0:
            return 'none'
        
        # Get adaptive thresholds
        extensive_thresh = self.get_threshold('extensive_depth_score', category)
        substantial_thresh = self.get_threshold('substantial_depth_score', category)
        moderate_thresh = self.get_threshold('moderate_depth_score', category)
        minimal_thresh = self.get_threshold('minimal_depth_score', category)
        
        # Assess based on score + quantity
        if avg_score >= extensive_thresh and num_docs >= 7:
            return 'extensive'
        elif avg_score >= substantial_thresh and num_docs >= 5:
            return 'substantial'
        elif avg_score >= moderate_thresh and num_docs >= 3:
            return 'moderate'
        elif avg_score >= minimal_thresh or num_docs >= 2:
            return 'limited'
        else:
            return 'minimal'
    
    def clear_cache(self):
        """Clear threshold cache."""
        cache.delete(self.cache_key)


# ==================== KNOWLEDGE STATISTICS ENGINE ====================

class KnowledgeStatisticsEngine:
    """
    Enhanced statistics sampling with random vector queries.
    Provides accurate KB distribution metrics.
    """
    
    def __init__(self, namespace: Optional[str] = None):
        """
        Initialize statistics engine.
        
        Args:
            namespace: Pinecone namespace
        """
        self.namespace = namespace
        self.index_name = INDEX_NAME
        self.cache_timeout = 3600
        self.cache_key_category = "kb_category_stats_v3"
        self.cache_key_similarity = "kb_similarity_stats_v3"
    
    def get_category_statistics(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """Get category distribution statistics with enhanced sampling."""
        if not force_refresh:
            cached = cache.get(self.cache_key_category)
            if cached:
                return cached
        
        stats = self._sample_category_statistics()
        cache.set(self.cache_key_category, stats, self.cache_timeout)
        return stats
    
    def get_similarity_statistics(self, force_refresh: bool = False) -> Dict:
        """Get global similarity distribution statistics."""
        if not force_refresh:
            cached = cache.get(self.cache_key_similarity)
            if cached:
                return cached
        
        stats = self._sample_similarity_distribution()
        cache.set(self.cache_key_similarity, stats, self.cache_timeout)
        return stats
    
    def _sample_category_statistics(self, sample_size: int = 300) -> Dict[str, Dict]:
        """Sample KB to estimate category distribution."""
        try:
            # FIXED: Use helper function for langchain_community
            vector_store = get_pinecone_vectorstore(self.index_name, self.namespace)
            
            # Diverse sample queries for broad coverage
            sample_queries = [
                "learning education study",
                "analysis research methodology",
                "development implementation practice",
                "theory concepts fundamentals",
                "application practical examples",
                "advanced specialized topics",
                "introduction basics overview",
                "technical professional academic"
            ]
            
            category_data = defaultdict(lambda: {
                'document_count': 0,
                'similarities': [],
                'complexities': []
            })
            
            docs_per_query = sample_size // len(sample_queries)
            
            for query in sample_queries:
                try:
                    results = vector_store.similarity_search_with_score(
                        query=query,
                        k=docs_per_query
                    )
                    
                    for doc, score in results:
                        meta = SchemaHandler.standardize_metadata(doc.metadata)
                        category = meta['category']
                        
                        category_data[category]['document_count'] += 1
                        category_data[category]['similarities'].append(score)
                        
                        if meta['complexity'] > 0:
                            category_data[category]['complexities'].append(meta['complexity'])
                
                except Exception as e:
                    print(f"⚠️ Query sampling failed: {e}")
                    continue
            
            # Compute statistics per category
            stats = {}
            for category, data in category_data.items():
                sims = data['similarities']
                if sims:
                    stats[category] = {
                        'document_count': data['document_count'],
                        'avg_similarity': float(np.mean(sims)),
                        'std_similarity': float(np.std(sims)),
                        'p25_similarity': float(np.percentile(sims, 25)),
                        'p50_similarity': float(np.percentile(sims, 50)),
                        'p75_similarity': float(np.percentile(sims, 75)),
                        'p90_similarity': float(np.percentile(sims, 90)),
                        'avg_complexity': float(np.mean(data['complexities'])) if data['complexities'] else 5.0
                    }
            
            return stats if stats else self._default_category_stats()
            
        except Exception as e:
            print(f"❌ Category statistics sampling failed: {e}")
            import traceback
            traceback.print_exc()
            return self._default_category_stats()
    
    def _sample_similarity_distribution(self, sample_size: int = 500) -> Dict:
        """Sample similarity distribution for global statistics."""
        try:
            # FIXED: Use helper function for langchain_community
            vector_store = get_pinecone_vectorstore(self.index_name, self.namespace)
            
            all_similarities = []
            
            # Random diverse queries
            random_queries = [
                "study", "learn", "analyze", "understand", "develop",
                "research", "examine", "investigate", "explore", "discover"
            ]
            
            random.shuffle(random_queries)
            
            for query in random_queries[:5]:
                try:
                    results = vector_store.similarity_search_with_score(
                        query=query,
                        k=100
                    )
                    
                    all_similarities.extend([score for _, score in results])
                
                except Exception as e:
                    print(f"⚠️ Similarity sampling failed: {e}")
                    continue
            
            if not all_similarities:
                return self._default_similarity_stats()
            
            return {
                'total_samples': len(all_similarities),
                'mean': float(np.mean(all_similarities)),
                'std': float(np.std(all_similarities)),
                'median': float(np.median(all_similarities)),
                'p25': float(np.percentile(all_similarities, 25)),
                'p50': float(np.percentile(all_similarities, 50)),
                'p75': float(np.percentile(all_similarities, 75)),
                'p90': float(np.percentile(all_similarities, 90)),
                'p95': float(np.percentile(all_similarities, 95)),
                'min': float(np.min(all_similarities)),
                'max': float(np.max(all_similarities))
            }
            
        except Exception as e:
            print(f"❌ Similarity distribution sampling failed: {e}")
            return self._default_similarity_stats()
    
    def _default_category_stats(self) -> Dict:
        """Fallback category statistics."""
        defaults = {}
        for cat in ['mathematics', 'data_science', 'programming', 'science', 'business', 'finance', 'general']:
            defaults[cat] = {
                'document_count': 100,
                'avg_similarity': 0.65,
                'std_similarity': 0.15,
                'p25_similarity': 0.55,
                'p50_similarity': 0.67,
                'p75_similarity': 0.78,
                'p90_similarity': 0.85,
                'avg_complexity': 5.0
            }
        return defaults
    
    def _default_similarity_stats(self) -> Dict:
        """Fallback similarity statistics."""
        return {
            'total_samples': 0,
            'mean': 0.65,
            'std': 0.15,
            'median': 0.67,
            'p25': 0.55,
            'p50': 0.67,
            'p75': 0.78,
            'p90': 0.85,
            'p95': 0.90,
            'min': 0.30,
            'max': 0.95
        }
    
    def clear_cache(self):
        """Clear all statistics caches."""
        cache.delete(self.cache_key_category)
        cache.delete(self.cache_key_similarity)


# ==================== MAIN KNOWLEDGE GROUNDING ENGINE ====================

class KnowledgeGroundingEngine:
    """
    Production-ready knowledge grounding with dynamic systems.
    FIXED for langchain_community.vectorstores.Pinecone
    """
    
    def __init__(
        self, 
        namespace: Optional[str] = None, 
        use_cache: bool = True,
        thresholds_config: Optional[Dict] = None
    ):
        """
        Initialize engine with dynamic systems.
        
        Args:
            namespace: Pinecone namespace
            use_cache: Enable caching
            thresholds_config: Custom threshold overrides
        """
        # FIXED: Use helper function for langchain_community
        self.index_name = INDEX_NAME
        self.namespace = namespace
        self.vector_store = get_pinecone_vectorstore(self.index_name, namespace)
        self.use_cache = use_cache
        
        # Initialize dynamic systems
        self.schema_handler = SchemaHandler()
        self.calibration_engine = DynamicCalibrationEngine()
        self.threshold_engine = AdaptiveThresholdEngine(thresholds_config)
        self.stats_engine = KnowledgeStatisticsEngine(namespace)
        
        print(f"🔍 Knowledge Grounding Engine initialized (Dynamic Mode)")
        print(f"   Namespace: {namespace or 'default'}")
        print(f"   Cache: {'enabled' if use_cache else 'disabled'}")
        print(f"   Dynamic Calibration: ✓")
        print(f"   Adaptive Thresholds: ✓")
    
    def compute_knowledge_relevance(
        self, 
        text: str, 
        metadata: Dict,
        top_k: int = 10,
        category_aware: bool = True
    ) -> Dict:
        """
        Compute knowledge-grounded relevance with dynamic systems.
        
        Args:
            text: Content to analyze
            metadata: Document metadata (flexible schema)
            top_k: Number of similar documents
            category_aware: Use category-aware normalization
        
        Returns:
            Comprehensive relevance assessment
        """
        try:
            # Validate input
            if not text or len(text) < 20:
                print("⚠️ Text too short for KB grounding")
                return self._default_score("insufficient_text")
            
            # Standardize metadata
            std_metadata = self.schema_handler.standardize_metadata(metadata)
            category = std_metadata['category']
            
            print(f"   Querying KB: {std_metadata['topic'][:50]}...")
            print(f"   Category: {category}")
            
            # Get adaptive threshold
            min_threshold = self.threshold_engine.get_threshold('min_similarity', category)
            
            # Query knowledge base
            similar_docs = self._safe_vector_search(
                text=text,
                category=category,
                top_k=top_k,
                min_threshold=min_threshold
            )
            
            if not similar_docs:
                print("   No similar documents found")
                return self._default_score("no_matches")
            
            print(f"   Found {len(similar_docs)} documents")
            
            # Process results
            processed_docs = []
            for doc, score in similar_docs:
                doc_meta = self.schema_handler.standardize_metadata(doc.metadata)
                processed_docs.append((doc_meta, score))
            
            # Compute relevance
            result = self._compute_relevance_scores(
                processed_docs=processed_docs,
                target_category=category,
                category_aware=category_aware
            )
            
            # Add context
            result['context'] = self._extract_context(processed_docs, std_metadata)
            result['target_category'] = category
            result['timestamp'] = datetime.now().isoformat()
            
            print(f"   ✅ KR: {result['knowledge_relevance_score']:.3f} (conf: {result['confidence']:.3f})")
            
            return result
            
        except Exception as e:
            print(f"❌ KB grounding failed: {e}")
            import traceback
            traceback.print_exc()
            return self._default_score("error", error=str(e))
    
    def _safe_vector_search(
        self,
        text: str,
        category: str,
        top_k: int,
        min_threshold: float
    ) -> List[Tuple]:
        """Safe vector search with filtering."""
        try:
            # Execute search
            results = self.vector_store.similarity_search_with_score(
                query=text,
                k=top_k * 2  # Fetch extra for filtering
            )
            
            # Filter by threshold
            filtered = [
                (doc, score) for doc, score in results
                if score >= min_threshold
            ]
            
            # Return top_k after filtering
            return filtered[:top_k]
            
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return []
    
    def _compute_relevance_scores(
        self,
        processed_docs: List[Tuple[Dict, float]],
        target_category: str,
        category_aware: bool
    ) -> Dict:
        """Compute comprehensive relevance scores with dynamic calibration."""
        if not processed_docs:
            return self._default_score("no_matches")
        
        # Extract data
        raw_scores = [score for _, score in processed_docs]
        categories = [meta['category'] for meta, _ in processed_docs]
        
        # Normalize scores
        if category_aware:
            normalized_score = self._normalize_with_category_context(
                scores=raw_scores,
                categories=categories,
                target_category=target_category
            )
        else:
            normalized_score = self._normalize_global(raw_scores)
        
        # Apply dynamic calibration
        calibrated_score = self.calibration_engine.apply_calibration(
            score=normalized_score,
            category=target_category
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            scores=raw_scores,
            categories=categories,
            target_category=target_category
        )
        
        # Assess knowledge depth using adaptive thresholds
        avg_score = np.mean(raw_scores)
        knowledge_depth = self.threshold_engine.assess_knowledge_depth(
            avg_score=avg_score,
            num_docs=len(processed_docs),
            category=target_category
        )
        
        return {
            "knowledge_relevance_score": round(calibrated_score, 4),
            "normalized_score": round(normalized_score, 4),
            "confidence": round(confidence, 4),
            "knowledge_depth": knowledge_depth,
            "documents_found": len(processed_docs),
            "top_similarity": round(max(raw_scores), 4),
            "mean_similarity": round(np.mean(raw_scores), 4),
            "median_similarity": round(np.median(raw_scores), 4),
            "category_coverage": len(set(categories)),
            "dominant_kb_category": max(set(categories), key=categories.count) if categories else "unknown",
            "raw_scores": [round(s, 4) for s in raw_scores[:3]]
        }
    
    def _normalize_with_category_context(
        self,
        scores: List[float],
        categories: List[str],
        target_category: str
    ) -> float:
        """Category-aware normalization with inverse frequency weighting."""
        if not scores:
            return 0.5
        
        # Get dynamic category statistics
        category_stats = self.stats_engine.get_category_statistics()
        
        # Calculate total documents
        total_docs = sum(cat_info.get('document_count', 0) for cat_info in category_stats.values())
        if total_docs == 0:
            total_docs = 1000  # Fallback
        
        # Calculate weighted scores
        weighted_scores = []
        for score, cat in zip(scores, categories):
            cat_info = category_stats.get(cat, {})
            cat_count = cat_info.get('document_count', 100)
            
            # Inverse document frequency weight
            idf_weight = np.log((total_docs + 1) / (cat_count + 1)) + 1
            
            # Category match bonus
            category_bonus = 1.5 if cat == target_category else 1.0
            
            # Combined weight
            final_weight = idf_weight * category_bonus
            weighted_scores.append(score * final_weight)
        
        if not weighted_scores:
            return 0.5
        
        # Percentile-based normalization for robustness
        sorted_scores = sorted(weighted_scores)
        n = len(sorted_scores)
        
        if n == 1:
            return float(np.clip(sorted_scores[0], 0, 1))
        
        # Adaptive scaling using percentiles
        median = np.median(sorted_scores)
        p75 = np.percentile(sorted_scores, 75)
        p90 = np.percentile(sorted_scores, 90)
        max_score = max(sorted_scores)
        
        # Multi-scale normalization
        if max_score <= median:
            normalized = 0.5
        elif median < max_score <= p75:
            normalized = 0.5 + (max_score - median) / (p75 - median + 0.01) * 0.25
        elif p75 < max_score <= p90:
            normalized = 0.75 + (max_score - p75) / (p90 - p75 + 0.01) * 0.15
        else:
            normalized = 0.9 + (max_score - p90) / (max(sorted_scores) - p90 + 0.01) * 0.1
        
        return float(np.clip(normalized, 0, 1))
    
    def _normalize_global(self, scores: List[float]) -> float:
        """Global normalization using adaptive statistics."""
        if not scores:
            return 0.5
        
        # Get dynamic global statistics
        global_stats = self.stats_engine.get_similarity_statistics()
        
        max_score = max(scores)
        global_mean = global_stats.get('mean', 0.65)
        global_std = global_stats.get('std', 0.15)
        
        # Z-score normalization with sigmoid
        if global_std > 0:
            z_score = (max_score - global_mean) / global_std
            normalized = 1 / (1 + np.exp(-z_score))
        else:
            normalized = 0.5
        
        return float(np.clip(normalized, 0, 1))
    
    def _calculate_confidence(
        self,
        scores: List[float],
        categories: List[str],
        target_category: str
    ) -> float:
        """Calculate confidence in relevance score."""
        if not scores:
            return 0.0
        
        # Factor 1: Document quantity
        doc_count = len(scores)
        doc_factor = min(doc_count / 10.0, 1.0) ** 0.7
        
        # Factor 2: Score quality
        mean_score = np.mean(scores)
        score_factor = mean_score
        
        # Factor 3: Score consistency
        if len(scores) > 1:
            std = np.std(scores)
            consistency_factor = 1 - min(std / 0.3, 1.0)
        else:
            consistency_factor = 0.5
        
        # Factor 4: Category relevance
        same_category_count = sum(1 for cat in categories if cat == target_category)
        category_factor = same_category_count / len(categories) if categories else 0
        
        # Factor 5: Score distribution quality
        if len(scores) >= 3:
            score_range = max(scores) - min(scores)
            distribution_factor = 1 - min(score_range / 0.5, 1.0)
        else:
            distribution_factor = 0.5
        
        # Weighted combination
        confidence = (
            0.20 * doc_factor +
            0.30 * score_factor +
            0.20 * consistency_factor +
            0.20 * category_factor +
            0.10 * distribution_factor
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def _extract_context(
        self,
        processed_docs: List[Tuple[Dict, float]],
        target_metadata: Dict
    ) -> Dict:
        """Extract rich contextual insights from similar documents."""
        if not processed_docs:
            return {"note": "No context available"}
        
        # Collect statistics
        categories = defaultdict(int)
        source_types = defaultdict(int)
        topics = defaultdict(int)
        complexities = []
        files = []
        scores = [score for _, score in processed_docs]
        
        for meta, score in processed_docs:
            categories[meta['category']] += 1
            source_types[meta['source_type']] += 1
            topics[meta['topic']] += 1
            
            if meta['complexity'] > 0:
                complexities.append(meta['complexity'])
            
            if meta['file'] != 'unknown':
                files.append(meta['file'])
        
        # Build comprehensive context
        context = {
            "category_distribution": dict(categories),
            "source_type_distribution": dict(source_types),
            "dominant_category": max(categories, key=categories.get) if categories else "unknown",
            "dominant_source_type": max(source_types, key=source_types.get) if source_types else "unknown",
            "top_topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]),
            "complexity_stats": {
                "avg": round(np.mean(complexities), 2) if complexities else None,
                "min": min(complexities) if complexities else None,
                "max": max(complexities) if complexities else None,
                "median": round(np.median(complexities), 2) if complexities else None
            },
            "similarity_distribution": {
                "very_high (≥0.8)": sum(1 for s in scores if s >= 0.8),
                "high (0.7-0.8)": sum(1 for s in scores if 0.7 <= s < 0.8),
                "moderate (0.6-0.7)": sum(1 for s in scores if 0.6 <= s < 0.7),
                "low (0.5-0.6)": sum(1 for s in scores if 0.5 <= s < 0.6),
                "very_low (<0.5)": sum(1 for s in scores if s < 0.5)
            },
            "top_similar_sources": list(set(files))[:5],
            "coverage_metrics": {
                "unique_categories": len(categories),
                "unique_source_types": len(source_types),
                "unique_topics": len(topics)
            }
        }
        
        # Add intelligent interpretation
        target_complexity = target_metadata['complexity']
        if complexities:
            kb_avg_complexity = np.mean(complexities)
            complexity_diff = target_complexity - kb_avg_complexity
            
            if complexity_diff > 2:
                context['interpretation'] = (
                    f"This material (complexity {target_complexity}) is more advanced "
                    f"than similar KB content (avg {kb_avg_complexity:.1f}). "
                    f"Represents opportunity for advanced learning or knowledge gap."
                )
            elif complexity_diff < -2:
                context['interpretation'] = (
                    f"This material (complexity {target_complexity}) is more introductory "
                    f"than KB content (avg {kb_avg_complexity:.1f}). "
                    f"Good for building prerequisites or review."
                )
            else:
                context['interpretation'] = (
                    f"Complexity ({target_complexity}) aligns with KB (avg {kb_avg_complexity:.1f}). "
                    f"Standard priority relative to existing knowledge."
                )
        
        # Source type recommendations
        target_source = target_metadata['source_type']
        dominant_source = context['dominant_source_type']
        
        if target_source == 'assignment' and dominant_source in ['textbook', 'course_material']:
            context['recommendation'] = "Assignment complements existing course materials. Practice opportunity."
        elif target_source == 'exam_prep':
            context['recommendation'] = "Exam preparation material. Critical priority for assessment readiness."
        elif target_source == 'research_paper' and dominant_source == 'textbook':
            context['recommendation'] = "Research extends textbook foundations. Advanced learning opportunity."
        elif target_source == 'textbook' and dominant_source == 'research_paper':
            context['recommendation'] = "Textbook provides foundational context. Essential prerequisite material."
        else:
            context['recommendation'] = f"Standard {target_source} material with good KB coverage."
        
        return context
    
    def _default_score(self, reason: str = "unknown", error: str = None) -> Dict:
        """Robust default score with reason tracking."""
        return {
            "knowledge_relevance_score": 0.50,
            "normalized_score": 0.50,
            "confidence": 0.0,
            "knowledge_depth": "none",
            "documents_found": 0,
            "top_similarity": 0.0,
            "mean_similarity": 0.0,
            "median_similarity": 0.0,
            "category_coverage": 0,
            "dominant_kb_category": "unknown",
            "context": {
                "reason": reason,
                "error": error,
                "note": "Using default neutral score due to insufficient data"
            },
            "raw_scores": [],
            "timestamp": datetime.now().isoformat()
        }
    
    def batch_compute_relevance(
        self,
        documents: List[Dict],
        top_k: int = 10,
        category_aware: bool = True
    ) -> List[Dict]:
        """Batch processing for multiple documents with progress tracking."""
        print(f"📊 Batch processing {len(documents)} documents...")
        
        results = []
        for idx, doc in enumerate(documents, 1):
            if idx % 10 == 0:
                print(f"   Progress: {idx}/{len(documents)}")
            
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            relevance = self.compute_knowledge_relevance(
                text=text,
                metadata=metadata,
                top_k=top_k,
                category_aware=category_aware
            )
            
            results.append({
                **doc,
                'knowledge_grounding': relevance
            })
        
        print(f"✅ Batch complete: {len(results)} documents processed")
        return results
    
    def refresh_all_caches(self):
        """Force refresh of all caches in the system."""
        print("🔄 Refreshing all knowledge base caches...")
        
        # Refresh calibration
        self.calibration_engine.clear_cache()
        
        # Refresh thresholds
        self.threshold_engine.clear_cache()
        
        # Refresh statistics
        self.stats_engine.clear_cache()
        
        # Refresh category system
        SchemaHandler.refresh_category_system()
        
        print("✅ All caches refreshed")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status and statistics."""
        try:
            # Category statistics
            category_stats = self.stats_engine.get_category_statistics()
            
            # Similarity statistics
            similarity_stats = self.stats_engine.get_similarity_statistics()
            
            # Category system info
            all_categories = SchemaHandler.get_all_discovered_categories()
            
            # Calibration info
            sample_calibration = self.calibration_engine.get_domain_parameters('mathematics')
            
            # Threshold info
            sample_threshold = self.threshold_engine.get_threshold('min_similarity')
            
            return {
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "category_system": {
                    "total_categories": len(all_categories),
                    "categories": all_categories[:10],  # Sample
                    "dynamic_learning": True
                },
                "statistics": {
                    "categories_tracked": len(category_stats),
                    "total_samples": similarity_stats.get('total_samples', 0),
                    "mean_similarity": similarity_stats.get('mean', 0),
                    "std_similarity": similarity_stats.get('std', 0)
                },
                "calibration": {
                    "dynamic": True,
                    "sample_params": sample_calibration
                },
                "thresholds": {
                    "adaptive": True,
                    "sample_threshold": sample_threshold
                },
                "cache_status": {
                    "enabled": self.use_cache,
                    "namespace": self.namespace or "default"
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# ==================== INTEGRATION HELPERS ====================

def enhance_task_with_knowledge(
    task_analysis: Dict,
    text_content: str,
    blend_weight: float = 0.3,
    min_confidence: float = 0.3,
    enable_knowledge_boost: bool = True,
    thresholds_config: Optional[Dict] = None
) -> Dict:
    """
    Main integration point: Enhance task analysis with knowledge grounding.
    
    Args:
        task_analysis: Existing task analysis
        text_content: Document content
        blend_weight: KB influence weight (0-1)
        min_confidence: Minimum confidence threshold
        enable_knowledge_boost: Enable priority boost for knowledge gaps
        thresholds_config: Custom threshold configuration
    
    Returns:
        Enhanced task analysis with KB grounding
    """
    try:
        # Initialize engine with custom config
        engine = KnowledgeGroundingEngine(
            namespace="knowledge_base",
            use_cache=True,
            thresholds_config=thresholds_config
        )
        
        # Prepare metadata
        metadata = {
            'category': task_analysis.get('category', 'general'),
            'complexity': task_analysis.get('complexity', 5),
            'pages': task_analysis.get('pages', 0),
            'urgency_score': task_analysis.get('urgency_score', 5),
            'is_foundational': task_analysis.get('is_foundational', False),
            'source_type': task_analysis.get('source_type', 'unknown'),
            'topic': task_analysis.get('task', 'unknown')
        }
        
        # Compute KB relevance
        kb_result = engine.compute_knowledge_relevance(
            text=text_content,
            metadata=metadata,
            top_k=10,
            category_aware=True
        )
        
        # Add to task analysis
        task_analysis['knowledge_grounding'] = kb_result
        
        # Extract metrics
        kr_score = kb_result['knowledge_relevance_score']
        confidence = kb_result['confidence']
        kb_depth = kb_result['knowledge_depth']
        
        # Original score
        original_score = task_analysis.get('preferred_score', 5.0)
        
        # Apply KB-based adjustment
        if confidence >= min_confidence:
            # Dynamic blending
            effective_blend = blend_weight * (1 + confidence * 0.5)
            effective_blend = min(effective_blend, 0.5)  # Cap at 50%
            
            # Blend scores
            blended_score = (
                original_score * (1 - effective_blend) +
                kr_score * 10 * effective_blend
            )
            
            # Knowledge gap boost
            if enable_knowledge_boost and kb_depth in ['minimal', 'limited', 'none']:
                gap_boost = 0.15 * (1 - (kr_score * confidence))
                blended_score += gap_boost
                task_analysis['knowledge_gap_boost'] = round(gap_boost, 3)
            
            task_analysis['knowledge_adjusted_score'] = round(blended_score, 2)
            task_analysis['adjustment_factor'] = round(effective_blend, 3)
            task_analysis['knowledge_influence'] = (
                "high" if effective_blend > 0.35 else
                "moderate" if effective_blend > 0.20 else
                "low"
            )
        else:
            # Low confidence - minimal adjustment
            task_analysis['knowledge_adjusted_score'] = original_score
            task_analysis['adjustment_factor'] = 0.0
            task_analysis['knowledge_influence'] = "minimal"
        
        # Generate reasoning context
        task_analysis['kb_reasoning_context'] = _generate_kb_reasoning_context(
            kb_result=kb_result,
            task_analysis=task_analysis,
            original_score=original_score
        )
        
        # Interpretation
        task_analysis['kb_interpretation'] = _interpret_kb_result(kb_result)
        
        print(f"   ✅ KB Enhancement: KR={kr_score:.3f}, Conf={confidence:.3f}, "
              f"Adjusted={task_analysis['knowledge_adjusted_score']:.2f}")
        
        return task_analysis
        
    except Exception as e:
        print(f"⚠️ KB enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Graceful fallback
        task_analysis['knowledge_grounding'] = {
            "error": str(e),
            "knowledge_relevance_score": 0.5,
            "confidence": 0.0
        }
        task_analysis['knowledge_adjusted_score'] = task_analysis.get('preferred_score', 5.0)
        task_analysis['kb_reasoning_context'] = {
            "status": "unavailable",
            "note": "KB grounding unavailable"
        }
        
        return task_analysis


def _generate_kb_reasoning_context(
    kb_result: Dict,
    task_analysis: Dict,
    original_score: float
) -> Dict:
    """Generate structured reasoning context for LLM."""
    kr_score = kb_result['knowledge_relevance_score']
    confidence = kb_result['confidence']
    kb_depth = kb_result['knowledge_depth']
    context = kb_result.get('context', {})
    
    reasoning = {
        "knowledge_coverage": {
            "depth": kb_depth,
            "confidence": confidence,
            "documents_found": kb_result['documents_found'],
            "interpretation": None
        },
        "domain_context": {
            "target_category": kb_result.get('target_category', 'unknown'),
            "dominant_kb_category": kb_result['dominant_kb_category'],
            "category_alignment": kb_result.get('target_category') == kb_result['dominant_kb_category']
        },
        "similarity_metrics": {
            "relevance_score": kr_score,
            "top_similarity": kb_result['top_similarity'],
            "mean_similarity": kb_result['mean_similarity']
        },
        "priority_rationale": None,
        "learning_recommendation": None
    }
    
    # Interpret knowledge coverage
    if kb_depth == 'extensive':
        reasoning['knowledge_coverage']['interpretation'] = (
            "Extensively covered in KB - abundant reference materials available."
        )
        reasoning['priority_rationale'] = (
            "May receive standard/lower priority due to abundant existing resources."
        )
        reasoning['learning_recommendation'] = (
            "Leverage KB materials for self-study. Focus on advanced applications."
        )
    
    elif kb_depth == 'substantial':
        reasoning['knowledge_coverage']['interpretation'] = (
            "Good KB coverage provides solid foundation."
        )
        reasoning['priority_rationale'] = (
            "Standard priority - balanced against other factors."
        )
        reasoning['learning_recommendation'] = (
            "Use KB materials as supplementary resources."
        )
    
    elif kb_depth in ['moderate', 'limited']:
        reasoning['knowledge_coverage']['interpretation'] = (
            "Limited KB coverage suggests less-explored topic."
        )
        reasoning['priority_rationale'] = (
            "May receive priority boost to fill knowledge gaps."
        )
        reasoning['learning_recommendation'] = (
            "Valuable material for expanding knowledge base."
        )
    
    elif kb_depth in ['minimal', 'none']:
        reasoning['knowledge_coverage']['interpretation'] = (
            "Minimal/no KB coverage - significant knowledge gap."
        )
        reasoning['priority_rationale'] = (
            "Priority boost applied due to knowledge gap."
        )
        reasoning['learning_recommendation'] = (
            "High-value learning opportunity in new area."
        )
    
    # Add contextual insights
    if context.get('interpretation'):
        reasoning['complexity_context'] = context['interpretation']
    
    if context.get('recommendation'):
        reasoning['source_context'] = context['recommendation']
    
    # Score adjustment explanation
    adjusted_score = task_analysis.get('knowledge_adjusted_score', original_score)
    if abs(adjusted_score - original_score) > 0.5:
        reasoning['adjustment_explanation'] = (
            f"Priority adjusted from {original_score:.2f} to {adjusted_score:.2f} "
            f"based on KB analysis (confidence: {confidence:.2f}). "
            f"{reasoning['priority_rationale']}"
        )
    else:
        reasoning['adjustment_explanation'] = (
            f"Priority maintained at {original_score:.2f}. "
            f"KB analysis confirms standard prioritization."
        )
    
    return reasoning


def _interpret_kb_result(kb_result: Dict) -> str:
    """Generate human-readable KB result interpretation."""
    kr_score = kb_result['knowledge_relevance_score']
    confidence = kb_result['confidence']
    kb_depth = kb_result['knowledge_depth']
    
    if confidence < 0.3:
        return "Insufficient KB data for reliable assessment"
    
    if kb_depth == 'extensive':
        if kr_score > 0.75:
            return "Extensively covered - abundant high-quality KB resources"
        else:
            return "Broad KB coverage but lower relevance"
    
    elif kb_depth == 'substantial':
        return "Well-covered in KB - good foundation materials exist"
    
    elif kb_depth in ['moderate', 'limited']:
        if kr_score > 0.65:
            return "Moderate KB coverage - some relevant materials available"
        else:
            return "Limited KB coverage - opportunity to expand knowledge"
    
    else:
        return "Minimal/no KB coverage - significant knowledge gap"


def refresh_knowledge_base_cache():
    """
    Refresh all KB caches after document ingestion.
    """
    engine = KnowledgeGroundingEngine(namespace="knowledge_base", use_cache=True)
    engine.refresh_all_caches()
    
    stats = engine.get_system_status()
    
    print(f"📊 Knowledge Base Status:")
    print(f"   Total categories: {stats['category_system']['total_categories']}")
    print(f"   Tracked categories: {stats['statistics']['categories_tracked']}")
    print(f"   Total samples: {stats['statistics']['total_samples']}")
    
    return stats


# ==================== TESTING ====================

def test_knowledge_grounding(sample_text: str = None, sample_metadata: Dict = None):
    """
    Test knowledge grounding system with dynamic components.
    """
    print("=" * 80)
    print("🧪 TESTING DYNAMIC KNOWLEDGE GROUNDING SYSTEM")
    print("=" * 80)
    
    engine = KnowledgeGroundingEngine(namespace="knowledge_base")
    
    # Default test data
    if not sample_text:
        sample_text = """
        Introduction to machine learning algorithms and their applications in data science.
        This course covers supervised learning, unsupervised learning, and deep learning fundamentals.
        Topics include linear regression, decision trees, neural networks, and model evaluation.
        """
    
    if not sample_metadata:
        sample_metadata = {
            'category': 'data_science',
            'complexity': 6,
            'pages': 250,
            'source_type': 'textbook',
            'topic': 'Machine Learning Fundamentals'
        }
    
    print(f"\n📝 Test Input:")
    print(f"   Topic: {sample_metadata.get('topic', 'Unknown')}")
    print(f"   Category: {sample_metadata.get('category', 'Unknown')}")
    print(f"   Text length: {len(sample_text)} characters")
    
    # Test KB grounding
    result = engine.compute_knowledge_relevance(
        text=sample_text,
        metadata=sample_metadata,
        top_k=10
    )
    
    print(f"\n📊 Knowledge Grounding Results:")
    print(f"   Relevance Score: {result['knowledge_relevance_score']:.3f}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Knowledge Depth: {result['knowledge_depth']}")
    print(f"   Documents Found: {result['documents_found']}")
    print(f"   Category Coverage: {result['category_coverage']}")
    
    # Test enhancement
    print(f"\n🔧 Testing Task Enhancement:")
    
    task_analysis = {
        'task': 'Machine Learning Textbook',
        'category': 'data_science',
        'complexity': 6,
        'pages': 250,
        'preferred_score': 7.5,
        'urgency_score': 6,
        'is_foundational': True
    }
    
    enhanced = enhance_task_with_knowledge(
        task_analysis=task_analysis,
        text_content=sample_text
    )
    
    print(f"   Original Score: {task_analysis['preferred_score']}")
    print(f"   Adjusted Score: {enhanced.get('knowledge_adjusted_score', 'N/A')}")
    print(f"   Adjustment Factor: {enhanced.get('adjustment_factor', 0):.3f}")
    print(f"   KB Influence: {enhanced.get('knowledge_influence', 'unknown')}")
    
    # Test system status
    print(f"\n📊 System Status:")
    status = engine.get_system_status()
    print(f"   Status: {status['status']}")
    print(f"   Total Categories: {status['category_system']['total_categories']}")
    print(f"   Dynamic Learning: {status['category_system']['dynamic_learning']}")
    print(f"   Categories Tracked: {status['statistics']['categories_tracked']}")
    
    print(f"\n✅ Test complete")
    print("=" * 80)
    
    return result, enhanced, status


def validate_schema_handling():
    """
    Test schema flexibility with various metadata structures.
    """
    print("=" * 80)
    print("🧪 VALIDATING SCHEMA FLEXIBILITY")
    print("=" * 80)
    
    handler = SchemaHandler()
    
    # Test cases with different schemas
    test_cases = [
        {
            "name": "Standard Schema",
            "metadata": {
                "category": "mathematics",
                "complexity": 7,
                "pages": 300,
                "source_type": "textbook"
            }
        },
        {
            "name": "Alternative Field Names",
            "metadata": {
                "subject": "data science",
                "difficulty": 8,
                "page_count": 150,
                "doc_type": "research_paper"
            }
        },
        {
            "name": "Missing Fields",
            "metadata": {
                "category": "programming"
            }
        },
        {
            "name": "Empty Metadata",
            "metadata": {}
        },
        {
            "name": "Non-standard Values",
            "metadata": {
                "category": "CS 101",
                "complexity": "hard",
                "pages": "unknown"
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n📋 Test: {test['name']}")
        print(f"   Input: {test['metadata']}")
        
        standardized = handler.standardize_metadata(test['metadata'])
        
        print(f"   Standardized:")
        print(f"      Category: {standardized['category']}")
        print(f"      Complexity: {standardized['complexity']}")
        print(f"      Pages: {standardized['pages']}")
        print(f"      Source Type: {standardized['source_type']}")
    
    print(f"\n✅ Schema validation complete")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_knowledge_grounding()
    validate_schema_handling()
    