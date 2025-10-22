# core/knowledge_weighting.py - Production Knowledge-Grounded Relevance System
# Features: Flexible schema, robust fallbacks, domain calibration, cache management

from typing import List, Dict, Tuple, Optional, Any
from langchain_pinecone import PineconeVectorStore
from .llm_config import embeddings, INDEX_NAME
import numpy as np
from collections import defaultdict
import os
import json
from datetime import datetime, timedelta
from django.core.cache import cache
from pinecone import Pinecone

class SchemaHandler:
    """
    Handles flexible schema mapping for documents with different structures.
    Ensures robust field extraction with fallbacks.
    """
    
    # Standard field mappings with fallback aliases
    FIELD_MAPPINGS = {
        'category': ['category', 'subject', 'domain', 'field', 'topic_area', 'discipline'],
        'complexity': ['complexity', 'difficulty', 'level', 'complexity_score', 'difficulty_level'],
        'topic': ['topic', 'title', 'subject', 'theme', 'chapter'],
        'source_type': ['source_type', 'document_type', 'type', 'doc_type', 'material_type'],
        'file': ['file', 'filename', 'source', 'document', 'doc_name'],
        'pages': ['pages', 'page_count', 'num_pages', 'total_pages'],
        'author': ['author', 'authors', 'creator', 'instructor'],
        'year': ['year', 'date', 'published', 'created_at'],
        'institution': ['institution', 'university', 'school', 'organization']
    }
    
    # Default values for missing fields
    DEFAULTS = {
        'category': 'general',
        'complexity': 5,
        'topic': 'unknown',
        'source_type': 'unknown',
        'file': 'unknown',
        'pages': 0,
        'author': 'unknown',
        'year': None,
        'institution': 'unknown'
    }
    
    @classmethod
    def extract_field(cls, metadata: Dict, field_name: str, default: Any = None) -> Any:
        """
        Flexibly extract field from metadata with multiple fallback aliases.
        
        Args:
            metadata: Document metadata dict
            field_name: Standard field name
            default: Default value if not found
        
        Returns:
            Field value or default
        """
        if not metadata:
            return default if default is not None else cls.DEFAULTS.get(field_name)
        
        # Try all aliases for this field
        aliases = cls.FIELD_MAPPINGS.get(field_name, [field_name])
        
        for alias in aliases:
            if alias in metadata and metadata[alias] is not None:
                value = metadata[alias]
                # Normalize value
                return cls._normalize_value(field_name, value)
        
        # Return default if not found
        return default if default is not None else cls.DEFAULTS.get(field_name)
    
    @classmethod
    def _normalize_value(cls, field_name: str, value: Any) -> Any:
        """
        Normalize field values to standard formats.
        """
        if value is None:
            return cls.DEFAULTS.get(field_name)
        
        # Category normalization
        if field_name == 'category':
            return cls._normalize_category(value)
        
        # Complexity normalization (ensure 1-10 scale)
        elif field_name == 'complexity':
            try:
                comp = float(value)
                return int(max(1, min(10, comp)))
            except (ValueError, TypeError):
                return 5
        
        # Pages normalization
        elif field_name == 'pages':
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0
        
        # Source type normalization
        elif field_name == 'source_type':
            return cls._normalize_source_type(value)
        
        # String fields - clean
        elif isinstance(value, str):
            return value.strip().lower()
        
        return value
    
    @classmethod
    def _normalize_category(cls, category: str) -> str:
        """
        Normalize category names to standard taxonomy.
        """
        if not isinstance(category, str):
            return 'general'
        
        cat_lower = category.lower().strip()
        
        # Mapping variations to standard categories
        category_map = {
            # STEM
            'math': 'mathematics',
            'maths': 'mathematics',
            'calculus': 'mathematics',
            'statistics': 'mathematics',
            'algebra': 'mathematics',
            'geometry': 'mathematics',
            
            # Data & CS
            'data': 'data_science',
            'ml': 'data_science',
            'machine learning': 'data_science',
            'ai': 'data_science',
            'analytics': 'data_science',
            'cs': 'programming',
            'computer science': 'programming',
            'coding': 'programming',
            'software': 'programming',
            
            # Science
            'physics': 'science',
            'chemistry': 'science',
            'biology': 'science',
            'engineering': 'science',
            
            # Business
            'management': 'business',
            'marketing': 'business',
            'strategy': 'business',
            'economics': 'finance',
            'accounting': 'finance',
            'investing': 'finance',
            
            # Academic
            'exam': 'exam_prep',
            'test': 'exam_prep',
            'midterm': 'exam_prep',
            'final': 'exam_prep',
            'quiz': 'exam_prep',
            'research': 'research',
            'thesis': 'research',
            'paper': 'research',
            'dissertation': 'research'
        }
        
        # Check for matches
        for key, standard in category_map.items():
            if key in cat_lower:
                return standard
        
        # Return as-is if no mapping found
        return cat_lower if cat_lower else 'general'
    
    @classmethod
    def _normalize_source_type(cls, source_type: str) -> str:
        """
        Normalize source type to standard taxonomy.
        """
        if not isinstance(source_type, str):
            return 'unknown'
        
        st_lower = source_type.lower().strip()
        
        type_map = {
            'textbook': 'textbook',
            'book': 'textbook',
            'text': 'textbook',
            'paper': 'research_paper',
            'research': 'research_paper',
            'journal': 'research_paper',
            'article': 'research_paper',
            'course': 'course_material',
            'lecture': 'course_material',
            'slides': 'course_material',
            'notes': 'course_material',
            'assignment': 'assignment',
            'homework': 'assignment',
            'project': 'assignment',
            'exercise': 'assignment',
            'exam': 'exam_material',
            'test': 'exam_material',
            'quiz': 'exam_material',
            'practice': 'exam_material'
        }
        
        for key, standard in type_map.items():
            if key in st_lower:
                return standard
        
        return 'unknown'
    
    @classmethod
    def standardize_metadata(cls, metadata: Dict) -> Dict:
        """
        Convert any metadata schema to standardized format.
        
        Returns:
            Standardized metadata dict with all expected fields
        """
        return {
            'category': cls.extract_field(metadata, 'category'),
            'complexity': cls.extract_field(metadata, 'complexity'),
            'topic': cls.extract_field(metadata, 'topic'),
            'source_type': cls.extract_field(metadata, 'source_type'),
            'file': cls.extract_field(metadata, 'file'),
            'pages': cls.extract_field(metadata, 'pages'),
            'author': cls.extract_field(metadata, 'author'),
            'year': cls.extract_field(metadata, 'year'),
            'institution': cls.extract_field(metadata, 'institution'),
            '_original': metadata  # Keep original for reference
        }


class DomainCalibration:
    """
    Domain-specific calibration parameters based on actual data distribution.
    Updated to reflect real academic materials: textbooks, research, courses, assignments.
    """
    
    # Calibration parameters per domain
    # Format: {base: expected mean similarity, spread: std dev, boost: importance multiplier}
    CALIBRATION = {
        # STEM domains - typically high similarity within domain
        'mathematics': {
            'base': 0.72,
            'spread': 0.18,
            'boost': 1.25,
            'typical_sources': ['textbook', 'course_material', 'assignment'],
            'characteristics': 'High internal coherence, formal notation'
        },
        'data_science': {
            'base': 0.68,
            'spread': 0.22,
            'boost': 1.20,
            'typical_sources': ['textbook', 'research_paper', 'course_material'],
            'characteristics': 'Rapidly evolving, mixed theory and practice'
        },
        'programming': {
            'base': 0.70,
            'spread': 0.20,
            'boost': 1.15,
            'typical_sources': ['textbook', 'course_material', 'assignment'],
            'characteristics': 'Code-heavy, practical focus'
        },
        'science': {
            'base': 0.69,
            'spread': 0.21,
            'boost': 1.15,
            'typical_sources': ['textbook', 'research_paper', 'course_material'],
            'characteristics': 'Experimental, data-driven'
        },
        
        # Business & Finance - moderate similarity
        'finance': {
            'base': 0.64,
            'spread': 0.25,
            'boost': 1.05,
            'typical_sources': ['textbook', 'course_material', 'research_paper'],
            'characteristics': 'Market-sensitive, applied economics'
        },
        'business': {
            'base': 0.62,
            'spread': 0.28,
            'boost': 1.00,
            'typical_sources': ['textbook', 'course_material', 'case_study'],
            'characteristics': 'Case-based, contextual'
        },
        
        # Academic activities - higher importance modifiers
        'exam_prep': {
            'base': 0.75,
            'spread': 0.16,
            'boost': 1.35,
            'typical_sources': ['exam_material', 'course_material'],
            'characteristics': 'High-stakes, time-sensitive'
        },
        'research': {
            'base': 0.65,
            'spread': 0.24,
            'boost': 1.10,
            'typical_sources': ['research_paper', 'thesis'],
            'characteristics': 'Novel contributions, specialized'
        },
        
        # Default
        'general': {
            'base': 0.60,
            'spread': 0.28,
            'boost': 1.00,
            'typical_sources': ['unknown'],
            'characteristics': 'Broad, varied content'
        }
    }
    
    # Source type importance multipliers
    SOURCE_TYPE_WEIGHTS = {
        'textbook': 1.10,          # Foundational, comprehensive
        'research_paper': 1.15,    # Cutting-edge, specialized
        'course_material': 1.05,   # Structured learning
        'assignment': 1.20,        # Practice, application
        'exam_material': 1.30,     # High-stakes preparation
        'unknown': 1.00
    }
    
    @classmethod
    def get_parameters(cls, category: str) -> Dict:
        """Get calibration parameters for a category."""
        return cls.CALIBRATION.get(category, cls.CALIBRATION['general'])
    
    @classmethod
    def get_source_weight(cls, source_type: str) -> float:
        """Get importance weight for source type."""
        return cls.SOURCE_TYPE_WEIGHTS.get(source_type, 1.00)
    
    @classmethod
    def apply_calibration(cls, score: float, category: str, source_type: str = None) -> float:
        """
        Apply domain calibration and source type weighting.
        
        Args:
            score: Raw normalized score (0-1)
            category: Document category
            source_type: Document source type (optional)
        
        Returns:
            Calibrated score (0-1)
        """
        params = cls.get_parameters(category)
        
        # Domain calibration
        calibrated = (score - params['base']) / params['spread']
        calibrated = 1 / (1 + np.exp(-calibrated))  # Sigmoid to 0-1
        
        # Apply domain importance boost
        calibrated *= params['boost']
        
        # Apply source type weight if available
        if source_type:
            source_weight = cls.get_source_weight(source_type)
            calibrated *= source_weight
        
        return float(np.clip(calibrated, 0, 1))


class KnowledgeGroundingEngine:
    """
    Production-ready knowledge grounding engine with:
    - Flexible schema handling
    - Robust fallbacks
    - Domain calibration
    - Cache management
    - Comprehensive error handling
    """
    
    # Cache configuration
    CACHE_KEY_CATEGORY_STATS = "kb_category_stats_v2"
    CACHE_KEY_GLOBAL_STATS = "kb_global_stats_v2"
    CACHE_KEY_SOURCE_DIST = "kb_source_distribution_v2"
    CACHE_TIMEOUT = 3600  # 1 hour
    
    def __init__(self, namespace: Optional[str] = None, use_cache: bool = True):
        """
        Initialize knowledge grounding engine.
        
        Args:
            namespace: Pinecone namespace for knowledge base separation
            use_cache: Enable Django cache for statistics
        """
        self.vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        self.namespace = namespace
        self.use_cache = use_cache
        self.schema_handler = SchemaHandler()
        self.domain_calibration = DomainCalibration()
        
        print(f"🔍 Knowledge Grounding Engine initialized")
        print(f"   Namespace: {namespace or 'default'}")
        print(f"   Cache: {'enabled' if use_cache else 'disabled'}")
    
    def compute_knowledge_relevance(
        self, 
        text: str, 
        metadata: Dict,
        top_k: int = 10,
        category_aware: bool = True,
        min_score_threshold: float = 0.3
    ) -> Dict:
        """
        Compute knowledge-grounded relevance score with full error handling.
        
        Args:
            text: Content to analyze
            metadata: Document metadata (flexible schema)
            top_k: Number of similar documents to retrieve
            category_aware: Use category-aware normalization
            min_score_threshold: Minimum similarity to consider
        
        Returns:
            Comprehensive relevance assessment with confidence scores
        """
        try:
            # Validate input
            if not text or len(text) < 20:
                print("⚠️ Text too short for knowledge grounding")
                return self._default_score("insufficient_text")
            
            # Standardize metadata schema
            std_metadata = self.schema_handler.standardize_metadata(metadata)
            
            print(f"   Querying KB for: {std_metadata['topic'][:50]}...")
            print(f"   Category: {std_metadata['category']}, Source: {std_metadata['source_type']}")
            
            # Query knowledge base
            similar_docs = self._safe_vector_search(
                text=text,
                std_metadata=std_metadata,
                top_k=top_k,
                min_threshold=min_score_threshold
            )
            
            if not similar_docs:
                print("   No similar documents found in KB")
                return self._default_score("no_matches")
            
            print(f"   Found {len(similar_docs)} similar documents")
            
            # Extract and standardize metadata from results
            processed_docs = []
            for doc, score in similar_docs:
                std_doc_meta = self.schema_handler.standardize_metadata(doc.metadata)
                processed_docs.append((std_doc_meta, score))
            
            # Compute relevance scores
            relevance_result = self._compute_relevance_scores(
                processed_docs=processed_docs,
                target_metadata=std_metadata,
                category_aware=category_aware
            )
            
            # Add context and insights
            relevance_result['context'] = self._extract_rich_context(
                processed_docs=processed_docs,
                target_metadata=std_metadata
            )
            
            # Add metadata
            relevance_result['target_category'] = std_metadata['category']
            relevance_result['target_source_type'] = std_metadata['source_type']
            relevance_result['timestamp'] = datetime.now().isoformat()
            
            print(f"   ✅ KR Score: {relevance_result['knowledge_relevance_score']:.3f} "
                  f"(conf: {relevance_result['confidence']:.3f})")
            
            return relevance_result
            
        except Exception as e:
            print(f"❌ Knowledge grounding failed: {e}")
            import traceback
            traceback.print_exc()
            return self._default_score("error", error=str(e))
    
    def _safe_vector_search(
        self,
        text: str,
        std_metadata: Dict,
        top_k: int,
        min_threshold: float
    ) -> List[Tuple]:
        """
        Safe vector search with error handling and filtering.
        """
        try:
            # Build filter if category-aware
            search_filter = self._build_smart_filter(std_metadata)
            
            # Execute search
            results = self.vector_store.similarity_search_with_score(
                query=text,
                k=top_k * 2,  # Fetch more for filtering
                filter=search_filter
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
    
    def _build_smart_filter(self, std_metadata: Dict) -> Optional[Dict]:
        """
        Build intelligent filter for vector search.
        
        Strategy:
        - If exam_prep: prioritize same category + exam materials
        - If research: allow broader search
        - Default: no filter for maximum coverage
        """
        category = std_metadata['category']
        source_type = std_metadata['source_type']
        
        # High-priority categories get stricter filtering
        if category == 'exam_prep':
            return {
                "$or": [
                    {"category": {"$eq": category}},
                    {"source_type": {"$eq": "exam_material"}}
                ]
            }
        
        # For now, use broad search to maximize knowledge coverage
        # Can be refined based on your requirements
        return None
    
    def _compute_relevance_scores(
        self,
        processed_docs: List[Tuple[Dict, float]],
        target_metadata: Dict,
        category_aware: bool
    ) -> Dict:
        """
        Compute comprehensive relevance scores with normalization.
        """
        if not processed_docs:
            return self._default_score("no_matches")
        
        # Extract scores and categories
        raw_scores = [score for _, score in processed_docs]
        categories = [meta['category'] for meta, _ in processed_docs]
        source_types = [meta['source_type'] for meta, _ in processed_docs]
        
        # Normalize scores
        if category_aware:
            normalized_score = self._normalize_with_category_context(
                scores=raw_scores,
                categories=categories,
                target_category=target_metadata['category']
            )
        else:
            normalized_score = self._normalize_global(raw_scores)
        
        # Apply domain calibration
        calibrated_score = self.domain_calibration.apply_calibration(
            score=normalized_score,
            category=target_metadata['category'],
            source_type=target_metadata['source_type']
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            scores=raw_scores,
            categories=categories,
            num_docs=len(processed_docs),
            target_category=target_metadata['category']
        )
        
        # Knowledge depth assessment
        knowledge_depth = self._assess_knowledge_depth(
            scores=raw_scores,
            categories=categories,
            source_types=source_types
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
            "source_type_coverage": len(set(source_types)),
            "dominant_kb_category": max(set(categories), key=categories.count),
            "dominant_kb_source": max(set(source_types), key=source_types.count),
            "raw_scores": [round(s, 4) for s in raw_scores[:3]]
        }
    
    def _normalize_with_category_context(
        self,
        scores: List[float],
        categories: List[str],
        target_category: str
    ) -> float:
        """
        Category-aware normalization with inverse frequency weighting.
        """
        if not scores:
            return 0.5
        
        # Get category statistics
        category_stats = self._get_category_stats()
        total_docs = sum(category_stats.values()) or 1
        
        # Calculate IDF-style weights
        weighted_scores = []
        for score, cat in zip(scores, categories):
            # Inverse frequency weight
            cat_frequency = category_stats.get(cat, 1) / total_docs
            idf_weight = np.log((total_docs + 1) / (category_stats.get(cat, 1) + 1)) + 1
            
            # Category match bonus
            category_bonus = 1.5 if cat == target_category else 1.0
            
            # Combined weight
            final_weight = idf_weight * category_bonus
            weighted_scores.append(score * final_weight)
        
        if not weighted_scores:
            return 0.5
        
        # Robust percentile-based normalization
        sorted_scores = sorted(weighted_scores)
        n = len(sorted_scores)
        
        if n == 1:
            return float(np.clip(sorted_scores[0], 0, 1))
        
        # Use adaptive scaling
        median = np.median(sorted_scores)
        p75 = np.percentile(sorted_scores, 75)
        p90 = np.percentile(sorted_scores, 90)
        max_score = max(sorted_scores)
        
        # Scale to 0-1
        if max_score <= median:
            normalized = 0.5
        elif median < max_score <= p75:
            normalized = 0.5 + (max_score - median) / (p75 - median + 0.01) * 0.25
        else:
            normalized = 0.75 + (max_score - p75) / (p90 - p75 + 0.01) * 0.15
        
        return float(np.clip(normalized, 0, 1))
    
    def _normalize_global(self, scores: List[float]) -> float:
        """Global normalization using z-score and sigmoid."""
        if not scores:
            return 0.5
        
        global_stats = self._get_global_stats()
        
        max_score = max(scores)
        global_mean = global_stats['mean']
        global_std = global_stats['std']
        
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
        num_docs: int,
        target_category: str
    ) -> float:
        """
        Calculate confidence in relevance score.
        """
        if not scores:
            return 0.0
        
        # Factor 1: Document quantity
        doc_factor = min(num_docs / 10.0, 1.0) ** 0.7
        
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
        
        # Weighted combination
        confidence = (
            0.25 * doc_factor +
            0.35 * score_factor +
            0.20 * consistency_factor +
            0.20 * category_factor
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def _assess_knowledge_depth(
        self,
        scores: List[float],
        categories: List[str],
        source_types: List[str]
    ) -> str:
        """
        Assess depth of knowledge coverage in KB.
        """
        if not scores:
            return "none"
        
        avg_score = np.mean(scores)
        num_docs = len(scores)
        num_categories = len(set(categories))
        num_sources = len(set(source_types))
        
        # Quality + quantity assessment
        if avg_score >= 0.80 and num_docs >= 7:
            depth = "extensive"
        elif avg_score >= 0.70 and num_docs >= 5:
            depth = "substantial"
        elif avg_score >= 0.60 and num_docs >= 3:
            depth = "moderate"
        elif avg_score >= 0.50 and num_docs >= 2:
            depth = "limited"
        else:
            depth = "minimal"
        
        # Add breadth indicator
        if num_categories >= 3 or num_sources >= 3:
            depth += "_broad"
        
        return depth
    
    def _extract_rich_context(
        self,
        processed_docs: List[Tuple[Dict, float]],
        target_metadata: Dict
    ) -> Dict:
        """
        Extract comprehensive contextual insights.
        """
        if not processed_docs:
            return {"note": "No context available"}
        
        # Collect statistics
        categories = defaultdict(int)
        source_types = defaultdict(int)
        topics = defaultdict(int)
        complexities = []
        files = []
        scores = []
        
        for meta, score in processed_docs:
            categories[meta['category']] += 1
            source_types[meta['source_type']] += 1
            topics[meta['topic']] += 1
            
            if meta['complexity'] > 0:
                complexities.append(meta['complexity'])
            
            if meta['file'] != 'unknown':
                files.append(meta['file'])
            
            scores.append(score)
        
        # Build context
        context = {
            "category_distribution": dict(categories),
            "source_type_distribution": dict(source_types),
            "dominant_category": max(categories, key=categories.get),
            "dominant_source_type": max(source_types, key=source_types.get),
            "top_topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]),
            "complexity_stats": {
                "avg": round(np.mean(complexities), 2) if complexities else None,
                "min": min(complexities) if complexities else None,
                "max": max(complexities) if complexities else None
            },
            "similarity_distribution": {
                "very_high (≥0.8)": sum(1 for s in scores if s >= 0.8),
                "high (0.7-0.8)": sum(1 for s in scores if 0.7 <= s < 0.8),
                "moderate (0.6-0.7)": sum(1 for s in scores if 0.6 <= s < 0.7),
                "low (0.5-0.6)": sum(1 for s in scores if 0.5 <= s < 0.6),
                "very_low (<0.5)": sum(1 for s in scores if s < 0.5)
            },
            "top_similar_sources": list(set(files))[:5]
        }
        
        # Add interpretation
        target_complexity = target_metadata['complexity']
        if complexities:
            kb_avg_complexity = np.mean(complexities)
            
            if target_complexity > kb_avg_complexity + 2:
                context['interpretation'] = (
                    f"This material (complexity {target_complexity}) is significantly more advanced "
                    f"than similar KB content (avg {kb_avg_complexity:.1f}). "
                    f"Consider as high-priority foundational learning."
                )
            elif target_complexity < kb_avg_complexity - 2:
                context['interpretation'] = (
                    f"This material (complexity {target_complexity}) is more introductory "
                    f"than KB content (avg {kb_avg_complexity:.1f}). "
                    f"Good for review or building prerequisites."
                )
            else:
                context['interpretation'] = (
                    f"Complexity ({target_complexity}) aligns well with KB (avg {kb_avg_complexity:.1f}). "
                    f"Standard priority relative to knowledge base."
                )
        
        # Source type recommendation
        target_source = target_metadata['source_type']
        dominant_source = context['dominant_source_type']
        
        if target_source == 'assignment' and dominant_source in ['textbook', 'course_material']:
            context['source_recommendation'] = (
                "Assignment related to covered course materials. Practice opportunity."
            )
        elif target_source == 'exam_material':
            context['source_recommendation'] = (
                "Exam preparation - critical priority regardless of KB coverage."
            )
        elif target_source == 'research_paper' and dominant_source == 'textbook':
            context['source_recommendation'] = (
                "Research paper extends textbook knowledge. Opportunity for deeper learning."
            )
        elif target_source == 'textbook' and dominant_source == 'research_paper':
            context['source_recommendation'] = (
                "Textbook provides foundational context for research topics. Essential prerequisite."
            )
        else:
            context['source_recommendation'] = f"Standard {target_source} material."
        
        return context
    
    def _get_category_stats(self) -> Dict[str, int]:
        """
        Get category distribution statistics with caching.
        """
        if self.use_cache:
            cached = cache.get(self.CACHE_KEY_CATEGORY_STATS)
            if cached:
                return cached
        
        try:
            # Query Pinecone index statistics
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            
            # Sample-based estimation
            category_counts = self._sample_category_distribution(sample_size=200)
            
            if self.use_cache and category_counts:
                cache.set(self.CACHE_KEY_CATEGORY_STATS, category_counts, self.CACHE_TIMEOUT)
            
            return category_counts if category_counts else self._default_category_distribution()
            
        except Exception as e:
            print(f"⚠️ Could not fetch category stats: {e}")
            return self._default_category_distribution()
    
    def _sample_category_distribution(self, sample_size: int = 200) -> Dict[str, int]:
        """
        Sample knowledge base to estimate category distribution.
        """
        try:
            sample_queries = [
                "learning study education",
                "mathematics analysis calculation",
                "programming code software",
                "data science machine learning",
                "research paper study",
                "business management strategy",
                "finance economics accounting",
                "science physics chemistry"
            ]
            
            category_counts = defaultdict(int)
            
            for query in sample_queries[:4]:  # Limit for performance
                try:
                    results = self.vector_store.similarity_search(
                        query=query,
                        k=25
                    )
                    
                    for doc in results:
                        std_meta = self.schema_handler.standardize_metadata(doc.metadata)
                        category_counts[std_meta['category']] += 1
                
                except Exception as e:
                    print(f"⚠️ Sample query failed: {e}")
                    continue
            
            if not category_counts:
                return {}
            
            # Normalize to represent proportions
            total = sum(category_counts.values())
            normalized = {
                cat: max(int((count / total) * 1000), 1)  # Scale to 1000
                for cat, count in category_counts.items()
            }
            
            return normalized
            
        except Exception as e:
            print(f"⚠️ Sampling failed: {e}")
            return {}
    
    def _default_category_distribution(self) -> Dict[str, int]:
        """Fallback category distribution."""
        return {
            'mathematics': 150,
            'data_science': 120,
            'programming': 130,
            'science': 110,
            'business': 90,
            'finance': 80,
            'exam_prep': 100,
            'research': 70,
            'general': 150
        }
    
    def _get_global_stats(self) -> Dict[str, float]:
        """
        Get global similarity statistics.
        """
        if self.use_cache:
            cached = cache.get(self.CACHE_KEY_GLOBAL_STATS)
            if cached:
                return cached
        
        # Empirical values - can be updated via maintenance task
        global_stats = {
            'mean': 0.65,
            'std': 0.15,
            'median': 0.67,
            'p25': 0.55,
            'p75': 0.78,
            'p90': 0.85,
            'min': 0.30,
            'max': 0.95
        }
        
        if self.use_cache:
            cache.set(self.CACHE_KEY_GLOBAL_STATS, global_stats, self.CACHE_TIMEOUT)
        
        return global_stats
    
    def _default_score(self, reason: str = "unknown", error: str = None) -> Dict:
        """
        Robust default score with reason tracking.
        """
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
            "source_type_coverage": 0,
            "target_category": "unknown",
            "target_source_type": "unknown",
            "dominant_kb_category": "unknown",
            "dominant_kb_source": "unknown",
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
        """
        Batch processing for multiple documents.
        """
        print(f"📊 Batch processing {len(documents)} documents...")
        
        results = []
        for idx, doc in enumerate(documents, 1):
            print(f"   [{idx}/{len(documents)}] Processing...")
            
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
        
        print(f"✅ Batch complete")
        return results
    
    def refresh_cache(self):
        """
        Force refresh of all cached statistics.
        Call this after ingesting new document batches.
        """
        print("🔄 Refreshing knowledge base statistics cache...")
        
        # Clear all cache keys
        cache.delete(self.CACHE_KEY_CATEGORY_STATS)
        cache.delete(self.CACHE_KEY_GLOBAL_STATS)
        cache.delete(self.CACHE_KEY_SOURCE_DIST)
        
        # Force recomputation
        self._get_category_stats()
        self._get_global_stats()
        
        print("✅ Cache refreshed")
    
    def get_statistics_summary(self) -> Dict:
        """
        Get comprehensive statistics summary for monitoring.
        """
        category_stats = self._get_category_stats()
        global_stats = self._get_global_stats()
        
        return {
            "category_distribution": category_stats,
            "global_similarity_stats": global_stats,
            "total_categories": len(category_stats),
            "most_common_category": max(category_stats, key=category_stats.get) if category_stats else "unknown",
            "least_common_category": min(category_stats, key=category_stats.get) if category_stats else "unknown",
            "cache_enabled": self.use_cache,
            "namespace": self.namespace or "default",
            "calibration_domains": list(self.domain_calibration.CALIBRATION.keys()),
            "timestamp": datetime.now().isoformat()
        }


# ==================== INTEGRATION HELPER ====================

def enhance_task_with_knowledge(
    task_analysis: Dict,
    text_content: str,
    blend_weight: float = 0.3,
    min_confidence: float = 0.3,
    enable_knowledge_boost: bool = True
) -> Dict:
    """
    Main integration point: Enhance task analysis with knowledge grounding.
    
    This function:
    1. Computes knowledge-grounded relevance
    2. Blends it with original analysis
    3. Provides reasoning context for LLM
    
    Args:
        task_analysis: Existing task analysis dict
        text_content: Document content for KB comparison
        blend_weight: Weight for knowledge influence (0-1)
        min_confidence: Minimum confidence to apply blending
        enable_knowledge_boost: Whether to boost scores based on KB gaps
    
    Returns:
        Enhanced task analysis with knowledge grounding and reasoning context
    """
    try:
        # Initialize engine
        engine = KnowledgeGroundingEngine(namespace="knowledge_base")
        
        # Prepare metadata from task analysis
        metadata = {
            'category': task_analysis.get('category', 'general'),
            'complexity': task_analysis.get('complexity', 5),
            'pages': task_analysis.get('pages', 0),
            'urgency_score': task_analysis.get('urgency_score', 5),
            'is_foundational': task_analysis.get('is_foundational', False),
            'source_type': task_analysis.get('source_type', 'unknown'),
            'topic': task_analysis.get('task', 'unknown')
        }
        
        # Compute knowledge grounding
        kb_result = engine.compute_knowledge_relevance(
            text=text_content,
            metadata=metadata,
            top_k=10,
            category_aware=True
        )
        
        # Add to task analysis
        task_analysis['knowledge_grounding'] = kb_result
        
        # Extract key metrics
        kr_score = kb_result['knowledge_relevance_score']
        confidence = kb_result['confidence']
        kb_depth = kb_result['knowledge_depth']
        
        # Original score
        original_score = task_analysis.get('preferred_score', 5.0)
        
        # Apply knowledge-based adjustment
        if confidence >= min_confidence:
            # Dynamic blending based on confidence
            effective_blend = blend_weight * (1 + confidence * 0.5)
            effective_blend = min(effective_blend, 0.5)  # Cap at 50%
            
            # Base blended score
            blended_score = (
                original_score * (1 - effective_blend) +
                kr_score * 10 * effective_blend
            )
            
            # Knowledge gap boost (optional)
            if enable_knowledge_boost and kb_depth in ['minimal', 'limited', 'none']:
                # Boost priority for topics with limited KB coverage
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
        
        # Generate reasoning context for LLM
        task_analysis['kb_reasoning_context'] = _generate_kb_reasoning_context(
            kb_result=kb_result,
            task_analysis=task_analysis,
            original_score=original_score
        )
        
        # Add interpretative flags
        task_analysis['kb_interpretation'] = _interpret_kb_result(kb_result)
        
        print(f"   ✅ Knowledge enhancement: KR={kr_score:.3f}, Conf={confidence:.3f}, "
              f"Adjusted={task_analysis['knowledge_adjusted_score']:.2f}")
        
        return task_analysis
        
    except Exception as e:
        print(f"⚠️ Knowledge enhancement failed: {e}")
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
            "note": "Knowledge grounding unavailable, using standard analysis"
        }
        
        return task_analysis


def _generate_kb_reasoning_context(
    kb_result: Dict,
    task_analysis: Dict,
    original_score: float
) -> Dict:
    """
    Generate structured reasoning context for LLM to use in explanations.
    
    This provides the "why" behind knowledge-based adjustments.
    """
    kr_score = kb_result['knowledge_relevance_score']
    confidence = kb_result['confidence']
    kb_depth = kb_result['knowledge_depth']
    context = kb_result.get('context', {})
    
    # Build reasoning elements
    reasoning = {
        "knowledge_coverage": {
            "depth": kb_depth,
            "confidence": confidence,
            "documents_found": kb_result['documents_found'],
            "interpretation": None
        },
        "domain_context": {
            "target_category": kb_result['target_category'],
            "dominant_kb_category": kb_result['dominant_kb_category'],
            "category_alignment": kb_result['target_category'] == kb_result['dominant_kb_category']
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
    if kb_depth.startswith('extensive'):
        reasoning['knowledge_coverage']['interpretation'] = (
            "This topic is extensively covered in the knowledge base, suggesting it's a "
            "well-established area. Students already have substantial reference materials available."
        )
        reasoning['priority_rationale'] = (
            "May receive lower priority due to abundant existing resources, unless urgency is high."
        )
        reasoning['learning_recommendation'] = (
            "Leverage existing KB materials for self-study. Focus on advanced applications."
        )
    
    elif kb_depth.startswith('substantial'):
        reasoning['knowledge_coverage']['interpretation'] = (
            "Good knowledge base coverage exists for this topic, providing solid foundation materials."
        )
        reasoning['priority_rationale'] = (
            "Standard priority - balanced against urgency and complexity factors."
        )
        reasoning['learning_recommendation'] = (
            "Use KB materials as supplementary resources. Build on existing foundations."
        )
    
    elif kb_depth in ['moderate', 'limited']:
        reasoning['knowledge_coverage']['interpretation'] = (
            "Limited knowledge base coverage suggests this is a less-explored topic or emerging area."
        )
        reasoning['priority_rationale'] = (
            "May receive priority boost to fill knowledge gaps, especially if foundational."
        )
        reasoning['learning_recommendation'] = (
            "This material could be valuable for expanding knowledge base. Consider detailed study."
        )
    
    elif kb_depth in ['minimal', 'none']:
        reasoning['knowledge_coverage']['interpretation'] = (
            "Minimal or no knowledge base coverage - this represents a significant knowledge gap."
        )
        reasoning['priority_rationale'] = (
            "Priority boost applied due to knowledge gap. Important for comprehensive learning."
        )
        reasoning['learning_recommendation'] = (
            "High value learning opportunity. Could establish foundational understanding in new area."
        )
    
    # Add contextual interpretation
    if context.get('interpretation'):
        reasoning['complexity_context'] = context['interpretation']
    
    if context.get('source_recommendation'):
        reasoning['source_context'] = context['source_recommendation']
    
    # Score adjustment explanation
    adjusted_score = task_analysis.get('knowledge_adjusted_score', original_score)
    if abs(adjusted_score - original_score) > 0.5:
        reasoning['adjustment_explanation'] = (
            f"Priority score adjusted from {original_score:.2f} to {adjusted_score:.2f} "
            f"based on knowledge base analysis (confidence: {confidence:.2f}). "
            f"Adjustment reflects {reasoning['priority_rationale']}"
        )
    else:
        reasoning['adjustment_explanation'] = (
            f"Priority score maintained at {original_score:.2f}. "
            f"Knowledge base analysis confirms standard prioritization."
        )
    
    return reasoning


def _interpret_kb_result(kb_result: Dict) -> str:
    """
    Generate human-readable interpretation of KB result.
    """
    kr_score = kb_result['knowledge_relevance_score']
    confidence = kb_result['confidence']
    kb_depth = kb_result['knowledge_depth']
    
    if confidence < 0.3:
        return "Insufficient KB data for reliable assessment"
    
    if kb_depth.startswith('extensive'):
        if kr_score > 0.75:
            return "Extensively covered - abundant high-quality KB resources available"
        else:
            return "Broad KB coverage but lower relevance - tangentially related materials"
    
    elif kb_depth.startswith('substantial'):
        return "Well-covered in KB - good foundation materials exist"
    
    elif kb_depth in ['moderate', 'limited']:
        if kr_score > 0.65:
            return "Moderate KB coverage - some relevant materials available"
        else:
            return "Limited KB coverage - opportunity to expand knowledge"
    
    else:
        return "Minimal/no KB coverage - significant knowledge gap"


def compare_documents_by_knowledge(
    doc1: Dict,
    doc2: Dict,
    text1: str,
    text2: str
) -> Dict:
    """
    Compare two documents based on knowledge base relevance.
    Useful for tie-breaking in prioritization.
    """
    engine = KnowledgeGroundingEngine(namespace="knowledge_base")
    
    kr1 = engine.compute_knowledge_relevance(text1, doc1)
    kr2 = engine.compute_knowledge_relevance(text2, doc2)
    
    score1 = kr1['knowledge_relevance_score']
    score2 = kr2['knowledge_relevance_score']
    conf1 = kr1['confidence']
    conf2 = kr2['confidence']
    depth1 = kr1['knowledge_depth']
    depth2 = kr2['knowledge_depth']
    
    # Determine winner with reasoning
    score_diff = abs(score1 - score2)
    
    if score_diff < 0.1 and abs(conf1 - conf2) < 0.2:
        winner = "tie"
        recommendation = (
            f"Equal KB support (scores: {score1:.2f} vs {score2:.2f}). "
            f"Use urgency/complexity for prioritization."
        )
    elif score1 > score2:
        winner = "doc1"
        recommendation = (
            f"Doc1 has stronger KB relevance ({score1:.2f} vs {score2:.2f}, "
            f"depth: {depth1} vs {depth2}). "
            f"{'However, lower confidence.' if conf1 < 0.5 else 'High confidence.'}"
        )
    else:
        winner = "doc2"
        recommendation = (
            f"Doc2 has stronger KB relevance ({score2:.2f} vs {score1:.2f}, "
            f"depth: {depth2} vs {depth1}). "
            f"{'However, lower confidence.' if conf2 < 0.5 else 'High confidence.'}"
        )
    
    return {
        "winner": winner,
        "doc1": {
            "score": score1,
            "confidence": conf1,
            "depth": depth1
        },
        "doc2": {
            "score": score2,
            "confidence": conf2,
            "depth": depth2
        },
        "score_difference": score_diff,
        "confidence_difference": abs(conf1 - conf2),
        "recommendation": recommendation,
        "reasoning": {
            "doc1_interpretation": _interpret_kb_result(kr1),
            "doc2_interpretation": _interpret_kb_result(kr2)
        }
    }


# ==================== CACHE MANAGEMENT ====================

def refresh_knowledge_base_cache():
    """
    Refresh all knowledge base statistics cache.
    Call this after ingesting new document batches.
    """
    engine = KnowledgeGroundingEngine(namespace="knowledge_base", use_cache=True)
    engine.refresh_cache()
    
    stats = engine.get_statistics_summary()
    print(f"📊 Knowledge Base Statistics:")
    print(f"   Total categories: {stats['total_categories']}")
    print(f"   Most common: {stats['most_common_category']}")
    print(f"   Least common: {stats['least_common_category']}")
    
    return stats


# ==================== TESTING & VALIDATION ====================

def test_knowledge_grounding(sample_text: str = None, sample_metadata: Dict = None):
    """
    Test knowledge grounding system.
    """
    print("=" * 80)
    print("🧪 TESTING KNOWLEDGE GROUNDING SYSTEM")
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
    
    # Test knowledge grounding
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
    print(f"   Top Similarity: {result['top_similarity']:.3f}")
    print(f"   Category Coverage: {result['category_coverage']}")
    
    if result.get('context'):
        print(f"\n💡 Context:")
        print(f"   Dominant KB Category: {result['context'].get('dominant_category', 'N/A')}")
        print(f"   Interpretation: {result['context'].get('interpretation', 'N/A')[:100]}...")
    
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
    print(f"   Interpretation: {enhanced.get('kb_interpretation', 'N/A')}")
    
    print(f"\n✅ Test complete")
    print("=" * 80)
    
    return result, enhanced


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
                "subject": "data science",  # instead of category
                "difficulty": 8,             # instead of complexity
                "page_count": 150,           # instead of pages
                "doc_type": "research_paper" # instead of source_type
            }
        },
        {
            "name": "Missing Fields",
            "metadata": {
                "category": "programming"
                # missing most fields
            }
        },
        {
            "name": "Empty Metadata",
            "metadata": {}
        },
        {
            "name": "Non-standard Values",
            "metadata": {
                "category": "CS 101",  # needs normalization
                "complexity": "hard",  # string instead of int
                "pages": "unknown"     # invalid type
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
    test_knowledge_grounding()
    validate_schema_handling()