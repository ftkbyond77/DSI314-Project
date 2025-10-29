# core/feedback_system.py - Feedback Collection & Reinforcement Learning

from django.db.models import Avg, Count, Q
from django.utils import timezone
from typing import Dict, List, Optional, Tuple
import json
import numpy as np

from .models import (
    PrioritizationFeedback, 
    ScoringModelAdjustment,
    StudyPlanHistory,
    UserAnalytics
)


class FeedbackCollector:
    """
    Handles collection and processing of user feedback on prioritization.
    """
    
    @staticmethod
    def collect_feedback(
        user,
        study_plan: StudyPlanHistory,
        feedback_type: str,
        rating_type: str,
        thumbs_up: Optional[bool] = None,
        star_rating: Optional[int] = None,
        feedback_text: str = "",
        task_name: str = "",
        task_priority_suggested: Optional[int] = None,
        aspects: Optional[Dict] = None,
        request = None
    ) -> PrioritizationFeedback:
        """
        Collect user feedback and store in database.
        
        Args:
            user: User object
            study_plan: StudyPlanHistory instance
            feedback_type: 'priority', 'schedule', 'content', 'overall'
            rating_type: 'thumbs', 'stars', 'detailed'
            thumbs_up: True/False for thumbs rating
            star_rating: 1-5 for star rating
            feedback_text: Optional text feedback
            task_name: Specific task being rated
            task_priority_suggested: User's suggested priority
            aspects: Dict of specific aspects rated
            request: Django request object for metadata
        
        Returns:
            PrioritizationFeedback instance
        """
        
        # Collect metadata
        context_metadata = {}
        
        if request:
            ip_address = FeedbackCollector._get_client_ip(request)
            user_agent = request.META.get('HTTP_USER_AGENT', '')
        else:
            ip_address = None
            user_agent = ''
        
        # Create feedback entry
        feedback = PrioritizationFeedback.objects.create(
            study_plan=study_plan,
            user=user,
            feedback_type=feedback_type,
            rating_type=rating_type,
            thumbs_up=thumbs_up,
            star_rating=star_rating,
            task_name=task_name,
            task_priority_suggested=task_priority_suggested,
            feedback_text=feedback_text,
            aspects=aspects or {},
            ip_address=ip_address,
            user_agent=user_agent,
            context_metadata=context_metadata
        )
        
        # Update study plan
        study_plan.feedback_provided = True
        study_plan.save(update_fields=['feedback_provided'])
        
        # Update user analytics
        analytics, created = UserAnalytics.objects.get_or_create(user=user)
        analytics.update_from_feedback(feedback)
        
        print(f"📝 Feedback collected: {feedback}")
        
        return feedback
    
    @staticmethod
    def _get_client_ip(request):
        """Extract client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class ReinforcementLearningEngine:
    """
    Analyzes feedback and adjusts scoring model weights.
    Implements simple reinforcement learning for continuous improvement.
    """
    
    # Default weights (baseline)
    DEFAULT_WEIGHTS = {
        'urgency': 0.30,
        'complexity': 0.25,
        'foundational': 0.20,
        'kb_weight': 0.15,
        'pages': 0.10
    }
    
    def __init__(self):
        self.learning_rate = 0.05  # How much to adjust per feedback batch
        self.min_feedback_count = 5  # Minimum feedback before adjustment
    
    def analyze_feedback_and_adjust(
        self,
        scope: str = 'global',
        user = None,
        category: str = "",
        force: bool = False
    ) -> List[ScoringModelAdjustment]:
        """
        Analyze recent feedback and propose/apply weight adjustments.
        
        Args:
            scope: 'global', 'user', or 'category'
            user: User object if scope='user'
            category: Category name if scope='category'
            force: If True, apply adjustments even with few feedback items
        
        Returns:
            List of ScoringModelAdjustment objects created
        """
        
        print(f"🔄 Analyzing feedback for {scope} scope...")
        
        # Get unprocessed feedback
        feedback_query = PrioritizationFeedback.objects.filter(processed=False)
        
        if scope == 'user' and user:
            feedback_query = feedback_query.filter(user=user)
        elif scope == 'category' and category:
            feedback_query = feedback_query.filter(
                study_plan__plan_json__contains=category
            )
        
        feedback_items = list(feedback_query)
        
        if len(feedback_items) < self.min_feedback_count and not force:
            print(f"⏸️  Insufficient feedback ({len(feedback_items)}/{self.min_feedback_count})")
            return []
        
        print(f"📊 Processing {len(feedback_items)} feedback items...")
        
        # Analyze feedback patterns
        analysis = self._analyze_feedback_patterns(feedback_items)
        
        # Calculate weight adjustments
        adjustments = self._calculate_adjustments(analysis, scope, user, category)
        
        # Create adjustment records
        adjustment_objects = []
        for adj in adjustments:
            adj_obj = ScoringModelAdjustment.objects.create(
                scope=scope,
                user=user if scope == 'user' else None,
                category=category if scope == 'category' else None,
                factor_name=adj['factor'],
                old_weight=adj['old_weight'],
                new_weight=adj['new_weight'],
                adjustment_delta=adj['delta'],
                reason=adj['reason'],
                feedback_count_basis=len(feedback_items),
                confidence_score=adj['confidence']
            )
            adjustment_objects.append(adj_obj)
            
            print(f"  ✅ {adj['factor']}: {adj['old_weight']:.3f} → {adj['new_weight']:.3f}")
        
        # Mark feedback as processed
        feedback_query.update(processed=True, processed_at=timezone.now())
        
        return adjustment_objects
    
    def _analyze_feedback_patterns(self, feedback_items: List) -> Dict:
        """
        Analyze patterns in feedback to determine what needs adjustment.
        """
        
        analysis = {
            'avg_rating': 0,
            'positive_count': 0,
            'negative_count': 0,
            'aspects': {},
            'common_complaints': [],
            'patterns': {}
        }
        
        # Calculate average rating
        star_ratings = [f.star_rating for f in feedback_items if f.star_rating]
        if star_ratings:
            analysis['avg_rating'] = sum(star_ratings) / len(star_ratings)
        
        thumbs = [f.thumbs_up for f in feedback_items if f.thumbs_up is not None]
        analysis['positive_count'] = sum(thumbs)
        analysis['negative_count'] = len(thumbs) - sum(thumbs)
        
        # Analyze specific aspects
        for feedback in feedback_items:
            if feedback.aspects:
                for aspect, rating in feedback.aspects.items():
                    if aspect not in analysis['aspects']:
                        analysis['aspects'][aspect] = []
                    analysis['aspects'][aspect].append(rating)
        
        # Calculate average aspect ratings
        for aspect, ratings in analysis['aspects'].items():
            analysis['aspects'][aspect] = sum(ratings) / len(ratings)
        
        # Identify patterns from feedback text
        feedback_texts = [f.feedback_text.lower() for f in feedback_items if f.feedback_text]
        
        # Simple keyword analysis
        keywords = {
            'urgency': ['urgent', 'deadline', 'time', 'rush'],
            'complexity': ['difficult', 'complex', 'hard', 'easy', 'simple'],
            'foundational': ['basic', 'fundamental', 'foundation', 'prerequisite'],
            'relevance': ['relevant', 'important', 'priority', 'needed']
        }
        
        for factor, words in keywords.items():
            count = sum(
                1 for text in feedback_texts 
                if any(word in text for word in words)
            )
            analysis['patterns'][factor] = count / len(feedback_texts) if feedback_texts else 0
        
        return analysis
    
    def _calculate_adjustments(
        self,
        analysis: Dict,
        scope: str,
        user,
        category: str
    ) -> List[Dict]:
        """
        Calculate specific weight adjustments based on feedback analysis.
        """
        
        adjustments = []
        
        # Get current weights
        current_weights = self._get_current_weights(scope, user, category)
        
        # Determine which factors need adjustment
        avg_rating = analysis['avg_rating']
        
        # If overall rating is low, adjust based on aspect ratings
        if avg_rating < 3.5:
            # Low ratings - need significant adjustments
            
            for factor, weight in current_weights.items():
                # Check if this factor was mentioned negatively
                pattern_score = analysis['patterns'].get(factor, 0)
                
                if pattern_score > 0.3:  # Factor mentioned frequently
                    # Adjust based on aspect ratings if available
                    aspect_rating = analysis['aspects'].get(factor, 3)
                    
                    if aspect_rating < 3:
                        # Decrease weight (users think it's over-weighted)
                        delta = -self.learning_rate * (3 - aspect_rating) / 3
                    else:
                        # Increase weight (users think it's under-weighted)
                        delta = self.learning_rate * (aspect_rating - 3) / 2
                    
                    new_weight = max(0.05, min(0.50, weight + delta))
                    
                    if abs(new_weight - weight) > 0.01:
                        adjustments.append({
                            'factor': factor,
                            'old_weight': weight,
                            'new_weight': new_weight,
                            'delta': delta,
                            'reason': f"Adjusted based on user feedback (avg rating: {avg_rating:.1f}, mentions: {pattern_score:.1%})",
                            'confidence': min(0.8, pattern_score)
                        })
        
        elif avg_rating >= 4.0:
            # High ratings - make small refinements
            for factor, weight in current_weights.items():
                aspect_rating = analysis['aspects'].get(factor)
                
                if aspect_rating and aspect_rating >= 4.5:
                    # Slightly increase successful factors
                    delta = self.learning_rate * 0.3
                    new_weight = min(0.50, weight + delta)
                    
                    adjustments.append({
                        'factor': factor,
                        'old_weight': weight,
                        'new_weight': new_weight,
                        'delta': delta,
                        'reason': f"Reinforcing successful factor (aspect rating: {aspect_rating:.1f})",
                        'confidence': 0.6
                    })
        
        # Normalize weights to sum to 1.0
        if adjustments:
            adjustments = self._normalize_weights(adjustments, current_weights)
        
        return adjustments
    
    def _get_current_weights(self, scope: str, user, category: str) -> Dict:
        """Get current weights for the given scope"""
        
        if scope == 'user' and user:
            # Check user analytics for personalized weights
            try:
                analytics = UserAnalytics.objects.get(user=user)
                if analytics.personalized_weights:
                    return analytics.personalized_weights
            except UserAnalytics.DoesNotExist:
                pass
        
        elif scope == 'category' and category:
            # Get latest category-specific adjustment
            latest = ScoringModelAdjustment.objects.filter(
                scope='category',
                category=category,
                applied=True
            ).order_by('-adjustment_date').first()
            
            if latest:
                # Reconstruct weights from adjustments
                weights = self.DEFAULT_WEIGHTS.copy()
                weights[latest.factor_name] = latest.new_weight
                return weights
        
        # Default: get latest global adjustments
        latest_adjustments = ScoringModelAdjustment.objects.filter(
            scope='global',
            applied=True
        ).order_by('-adjustment_date')[:5]
        
        weights = self.DEFAULT_WEIGHTS.copy()
        for adj in latest_adjustments:
            weights[adj.factor_name] = adj.new_weight
        
        return weights
    
    def _normalize_weights(self, adjustments: List[Dict], current_weights: Dict) -> List[Dict]:
        """Normalize adjusted weights to sum to 1.0"""
        
        # Build complete weight dict
        all_weights = current_weights.copy()
        for adj in adjustments:
            all_weights[adj['factor']] = adj['new_weight']
        
        # Normalize
        total = sum(all_weights.values())
        if total > 0:
            for factor in all_weights:
                all_weights[factor] /= total
        
        # Update adjustments
        for adj in adjustments:
            adj['new_weight'] = all_weights[adj['factor']]
            adj['delta'] = adj['new_weight'] - adj['old_weight']
        
        return adjustments
    
    def get_active_weights(self, user=None, category: str = "") -> Dict:
        """
        Get the currently active weights for scoring.
        Priority: user-specific > category-specific > global
        """
        
        # Try user-specific first
        if user:
            weights = self._get_current_weights('user', user, None)
            if weights != self.DEFAULT_WEIGHTS:
                return weights
        
        # Try category-specific
        if category:
            weights = self._get_current_weights('category', None, category)
            if weights != self.DEFAULT_WEIGHTS:
                return weights
        
        # Fallback to global
        return self._get_current_weights('global', None, None)
    
    def apply_adjustments(self, adjustments: List[ScoringModelAdjustment]) -> bool:
        """
        Apply a list of adjustments to the system.
        """
        
        for adj in adjustments:
            adj.apply_adjustment()
            
            # If user-specific, update UserAnalytics
            if adj.scope == 'user' and adj.user:
                analytics, created = UserAnalytics.objects.get_or_create(user=adj.user)
                
                if not analytics.personalized_weights:
                    analytics.personalized_weights = self.DEFAULT_WEIGHTS.copy()
                
                analytics.personalized_weights[adj.factor_name] = adj.new_weight
                analytics.save()
        
        print(f"✅ Applied {len(adjustments)} adjustments")
        return True


# ==================== CONVENIENCE FUNCTIONS ====================

def collect_user_feedback(
    user,
    study_plan_id: int,
    feedback_data: Dict,
    request = None
) -> PrioritizationFeedback:
    """
    Main entry point for collecting feedback.
    
    Args:
        user: User object
        study_plan_id: ID of StudyPlanHistory
        feedback_data: Dict with feedback details
        request: Django request object
    
    Returns:
        PrioritizationFeedback instance
    """
    
    from .models import StudyPlanHistory
    
    study_plan = StudyPlanHistory.objects.get(id=study_plan_id)
    
    collector = FeedbackCollector()
    return collector.collect_feedback(
        user=user,
        study_plan=study_plan,
        feedback_type=feedback_data.get('feedback_type', 'overall'),
        rating_type=feedback_data.get('rating_type', 'stars'),
        thumbs_up=feedback_data.get('thumbs_up'),
        star_rating=feedback_data.get('star_rating'),
        feedback_text=feedback_data.get('feedback_text', ''),
        task_name=feedback_data.get('task_name', ''),
        task_priority_suggested=feedback_data.get('task_priority_suggested'),
        aspects=feedback_data.get('aspects'),
        request=request
    )


def trigger_reinforcement_learning(scope: str = 'global', user=None) -> List[ScoringModelAdjustment]:
    """
    Trigger reinforcement learning to adjust model weights.
    Can be called periodically or after N feedback items.
    
    Args:
        scope: 'global', 'user', or 'category'
        user: User object if scope='user'
    
    Returns:
        List of adjustments made
    """
    
    engine = ReinforcementLearningEngine()
    adjustments = engine.analyze_feedback_and_adjust(scope=scope, user=user)
    
    if adjustments:
        engine.apply_adjustments(adjustments)
    
    return adjustments


def get_personalized_weights(user) -> Dict:
    """
    Get personalized scoring weights for a user.
    Falls back to global weights if no personalization exists.
    
    Args:
        user: User object
    
    Returns:
        Dict of factor weights
    """
    
    engine = ReinforcementLearningEngine()
    return engine.get_active_weights(user=user)