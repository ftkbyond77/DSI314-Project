#!/usr/bin/env python3
"""
feedback_debug.py - Debug and Testing Tool for Reinforcement Learning Feedback System

This script provides comprehensive testing and debugging capabilities for the 
Student Assistant's feedback collection and reinforcement learning system.

Features:
- Test feedback collection with various scenarios
- Simulate user feedback patterns
- Test reinforcement learning adjustments
- Validate model weight changes
- Performance testing and benchmarking
- Database integrity checks

Usage:
    python feedback_debug.py [command] [options]
    
Commands:
    test-collection     - Test feedback collection functionality
    simulate-feedback   - Generate simulated feedback data
    test-rl            - Test reinforcement learning engine
    validate-weights   - Validate current model weights
    benchmark          - Performance benchmarking
    check-db           - Database integrity checks
    reset-test-data    - Clean up test data
    full-test          - Run complete test suite
"""

import os
import sys
import django
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_assistant.settings')
django.setup()

from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import transaction

from core.models import (
    StudyPlanHistory, PrioritizationFeedback, ScoringModelAdjustment,
    UserAnalytics, Upload, Chunk
)
from core.feedback_system import (
    FeedbackCollector, ReinforcementLearningEngine,
    collect_user_feedback, trigger_reinforcement_learning,
    get_personalized_weights
)

User = get_user_model()


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    success: bool
    message: str
    duration: float
    data: Optional[Dict] = None


class FeedbackDebugger:
    """Main debugger class for feedback system testing"""
    
    def __init__(self):
        self.test_user = None
        self.test_plan = None
        self.test_results = []
        self.engine = ReinforcementLearningEngine()
        
    def setup_test_environment(self) -> bool:
        """Setup test user and study plan for testing"""
        try:
            print("🔧 Setting up test environment...")
            
            # Create or get test user
            self.test_user, created = User.objects.get_or_create(
                username='debug_test_user',
                defaults={
                    'email': 'debug@test.com',
                    'first_name': 'Debug',
                    'last_name': 'User'
                }
            )
            
            if created:
                print(f"✅ Created test user: {self.test_user.username}")
            else:
                print(f"✅ Using existing test user: {self.test_user.username}")
            
            # Create test study plan
            self.test_plan, created = StudyPlanHistory.objects.get_or_create(
                user=self.test_user,
                user_goal="Debug testing study plan",
                defaults={
                    'sort_method': 'hybrid',
                    'constraints': 'Test constraints',
                    'time_input': {'weeks': 2, 'days': 3},
                    'total_hours': 50.0,
                    'total_files': 1,
                    'total_pages': 10,
                    'total_chunks': 5,
                    'plan_json': [
                        {
                            'file': 'test_document.pdf',
                            'task': 'Read Chapter 1',
                            'priority': 8,
                            'reasoning': 'High priority foundational content'
                        },
                        {
                            'file': 'test_document.pdf', 
                            'task': 'Complete exercises',
                            'priority': 6,
                            'reasoning': 'Medium priority practice'
                        }
                    ],
                    'execution_time': 2.5,
                    'tool_calls': 10,
                    'status': 'active'
                }
            )
            
            if created:
                print(f"✅ Created test study plan: {self.test_plan.id}")
            else:
                print(f"✅ Using existing test study plan: {self.test_plan.id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up test environment: {e}")
            return False
    
    def test_feedback_collection(self) -> TestResult:
        """Test basic feedback collection functionality"""
        start_time = time.time()
        
        try:
            print("\n🧪 Testing feedback collection...")
            
            # Test different feedback types
            feedback_scenarios = [
                {
                    'name': 'Thumbs Up Feedback',
                    'data': {
                        'feedback_type': 'overall',
                        'rating_type': 'thumbs',
                        'thumbs_up': True
                    }
                },
                {
                    'name': 'Star Rating Feedback',
                    'data': {
                        'feedback_type': 'priority',
                        'rating_type': 'stars',
                        'star_rating': 4,
                        'task_name': 'Read Chapter 1'
                    }
                },
                {
                    'name': 'Detailed Feedback',
                    'data': {
                        'feedback_type': 'overall',
                        'rating_type': 'detailed',
                        'star_rating': 5,
                        'feedback_text': 'Great prioritization!',
                        'aspects': {
                            'urgency': 4,
                            'complexity': 5,
                            'relevance': 4,
                            'schedule_quality': 5
                        }
                    }
                }
            ]
            
            created_feedbacks = []
            
            for scenario in feedback_scenarios:
                print(f"  Testing: {scenario['name']}")
                
                feedback = collect_user_feedback(
                    user=self.test_user,
                    study_plan_id=self.test_plan.id,
                    feedback_data=scenario['data']
                )
                
                created_feedbacks.append(feedback)
                
                # Validate feedback was created correctly
                assert feedback.user == self.test_user
                assert feedback.study_plan == self.test_plan
                assert feedback.feedback_type == scenario['data']['feedback_type']
                assert feedback.rating_type == scenario['data']['rating_type']
                
                print(f"    ✅ Created feedback ID: {feedback.id}")
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Feedback Collection",
                success=True,
                message=f"Successfully created {len(created_feedbacks)} feedback entries",
                duration=duration,
                data={'feedback_count': len(created_feedbacks)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Feedback Collection",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def simulate_feedback_patterns(self, count: int = 20) -> TestResult:
        """Generate realistic feedback patterns for testing"""
        start_time = time.time()
        
        try:
            print(f"\n🎭 Simulating {count} feedback patterns...")
            
            # Define realistic feedback patterns
            patterns = [
                {
                    'name': 'Positive User',
                    'weight': 0.3,
                    'feedback_generator': self._generate_positive_feedback
                },
                {
                    'name': 'Critical User', 
                    'weight': 0.2,
                    'feedback_generator': self._generate_critical_feedback
                },
                {
                    'name': 'Neutral User',
                    'weight': 0.3,
                    'feedback_generator': self._generate_neutral_feedback
                },
                {
                    'name': 'Detailed User',
                    'weight': 0.2,
                    'feedback_generator': self._generate_detailed_feedback
                }
            ]
            
            created_count = 0
            
            for i in range(count):
                # Select pattern based on weights
                pattern = random.choices(
                    patterns, 
                    weights=[p['weight'] for p in patterns]
                )[0]
                
                feedback_data = pattern['feedback_generator']()
                
                feedback = collect_user_feedback(
                    user=self.test_user,
                    study_plan_id=self.test_plan.id,
                    feedback_data=feedback_data
                )
                
                created_count += 1
                
                if (i + 1) % 5 == 0:
                    print(f"  Created {i + 1}/{count} feedback entries...")
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Feedback Simulation",
                success=True,
                message=f"Generated {created_count} simulated feedback entries",
                duration=duration,
                data={'simulated_count': created_count}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Feedback Simulation",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def _generate_positive_feedback(self) -> Dict:
        """Generate positive feedback pattern"""
        return {
            'feedback_type': random.choice(['overall', 'priority', 'schedule']),
            'rating_type': random.choice(['thumbs', 'stars']),
            'thumbs_up': True if random.random() > 0.2 else None,
            'star_rating': random.randint(4, 5) if random.random() > 0.2 else None,
            'feedback_text': random.choice([
                'Excellent prioritization!',
                'Great study plan!',
                'Very helpful recommendations',
                'Perfect timing and organization'
            ])
        }
    
    def _generate_critical_feedback(self) -> Dict:
        """Generate critical feedback pattern"""
        return {
            'feedback_type': random.choice(['overall', 'priority', 'content']),
            'rating_type': random.choice(['thumbs', 'stars']),
            'thumbs_up': False if random.random() > 0.3 else None,
            'star_rating': random.randint(1, 2) if random.random() > 0.3 else None,
            'feedback_text': random.choice([
                'Priorities seem off',
                'Too much content at once',
                'Schedule is unrealistic',
                'Missing important topics'
            ])
        }
    
    def _generate_neutral_feedback(self) -> Dict:
        """Generate neutral feedback pattern"""
        return {
            'feedback_type': 'overall',
            'rating_type': 'stars',
            'star_rating': 3,
            'feedback_text': random.choice([
                'Decent plan',
                'Could be better',
                'Average recommendations',
                'Not bad, not great'
            ])
        }
    
    def _generate_detailed_feedback(self) -> Dict:
        """Generate detailed feedback pattern"""
        return {
            'feedback_type': 'overall',
            'rating_type': 'detailed',
            'star_rating': random.randint(3, 5),
            'feedback_text': random.choice([
                'Good overall structure but needs refinement',
                'Solid plan with room for improvement',
                'Well thought out with some minor issues'
            ]),
            'aspects': {
                'urgency': random.randint(2, 5),
                'complexity': random.randint(2, 5),
                'relevance': random.randint(3, 5),
                'schedule_quality': random.randint(2, 4)
            }
        }
    
    def test_reinforcement_learning(self) -> TestResult:
        """Test reinforcement learning engine"""
        start_time = time.time()
        
        try:
            print("\n🤖 Testing reinforcement learning engine...")
            
            # Get current weights
            current_weights = self.engine.get_active_weights()
            print(f"  Current weights: {current_weights}")
            
            # Test global scope adjustment
            print("  Testing global scope adjustment...")
            global_adjustments = self.engine.analyze_feedback_and_adjust(
                scope='global',
                force=True  # Force even with few feedback items
            )
            
            print(f"    Generated {len(global_adjustments)} global adjustments")
            
            # Test user-specific adjustment
            print("  Testing user-specific adjustment...")
            user_adjustments = self.engine.analyze_feedback_and_adjust(
                scope='user',
                user=self.test_user,
                force=True
            )
            
            print(f"    Generated {len(user_adjustments)} user adjustments")
            
            # Test weight application
            if global_adjustments:
                print("  Testing weight application...")
                self.engine.apply_adjustments(global_adjustments)
                print("    ✅ Weights applied successfully")
            
            # Get updated weights
            updated_weights = self.engine.get_active_weights()
            print(f"  Updated weights: {updated_weights}")
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Reinforcement Learning",
                success=True,
                message=f"Generated {len(global_adjustments)} global and {len(user_adjustments)} user adjustments",
                duration=duration,
                data={
                    'global_adjustments': len(global_adjustments),
                    'user_adjustments': len(user_adjustments),
                    'weight_changes': self._compare_weights(current_weights, updated_weights)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Reinforcement Learning",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def _compare_weights(self, old_weights: Dict, new_weights: Dict) -> Dict:
        """Compare weight changes"""
        changes = {}
        for factor in old_weights:
            if factor in new_weights:
                changes[factor] = {
                    'old': old_weights[factor],
                    'new': new_weights[factor],
                    'delta': new_weights[factor] - old_weights[factor]
                }
        return changes
    
    def validate_model_weights(self) -> TestResult:
        """Validate current model weights and adjustments"""
        start_time = time.time()
        
        try:
            print("\n🔍 Validating model weights...")
            
            # Check weight normalization
            weights = self.engine.get_active_weights()
            total_weight = sum(weights.values())
            
            print(f"  Total weight sum: {total_weight:.3f}")
            
            # Validate weight ranges
            weight_issues = []
            for factor, weight in weights.items():
                if weight < 0.05:
                    weight_issues.append(f"{factor}: {weight:.3f} (too low)")
                elif weight > 0.50:
                    weight_issues.append(f"{factor}: {weight:.3f} (too high)")
            
            # Check recent adjustments
            recent_adjustments = ScoringModelAdjustment.objects.filter(
                adjustment_date__gte=timezone.now() - timedelta(days=7)
            ).count()
            
            # Check unprocessed feedback
            unprocessed_feedback = PrioritizationFeedback.objects.filter(
                processed=False
            ).count()
            
            duration = time.time() - start_time
            
            is_valid = abs(total_weight - 1.0) < 0.01 and len(weight_issues) == 0
            
            return TestResult(
                test_name="Weight Validation",
                success=is_valid,
                message=f"Weights {'valid' if is_valid else 'invalid'} - Sum: {total_weight:.3f}, Issues: {len(weight_issues)}",
                duration=duration,
                data={
                    'total_weight': total_weight,
                    'weight_issues': weight_issues,
                    'recent_adjustments': recent_adjustments,
                    'unprocessed_feedback': unprocessed_feedback
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Weight Validation",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def benchmark_performance(self) -> TestResult:
        """Benchmark feedback system performance"""
        start_time = time.time()
        
        try:
            print("\n⚡ Running performance benchmarks...")
            
            # Benchmark feedback collection
            collection_times = []
            for i in range(10):
                start = time.time()
                collect_user_feedback(
                    user=self.test_user,
                    study_plan_id=self.test_plan.id,
                    feedback_data={
                        'feedback_type': 'overall',
                        'rating_type': 'stars',
                        'star_rating': 4
                    }
                )
                collection_times.append(time.time() - start)
            
            avg_collection_time = sum(collection_times) / len(collection_times)
            
            # Benchmark RL analysis
            rl_times = []
            for i in range(3):
                start = time.time()
                self.engine.analyze_feedback_and_adjust(scope='global', force=True)
                rl_times.append(time.time() - start)
            
            avg_rl_time = sum(rl_times) / len(rl_times)
            
            # Database query performance
            db_start = time.time()
            PrioritizationFeedback.objects.filter(user=self.test_user).count()
            db_time = time.time() - db_start
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Performance Benchmark",
                success=True,
                message=f"Collection: {avg_collection_time:.3f}s, RL: {avg_rl_time:.3f}s, DB: {db_time:.3f}s",
                duration=duration,
                data={
                    'avg_collection_time': avg_collection_time,
                    'avg_rl_time': avg_rl_time,
                    'db_query_time': db_time
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Performance Benchmark",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def check_database_integrity(self) -> TestResult:
        """Check database integrity and consistency"""
        start_time = time.time()
        
        try:
            print("\n🗄️ Checking database integrity...")
            
            issues = []
            
            # Check for orphaned feedback
            orphaned_feedback = PrioritizationFeedback.objects.filter(
                study_plan__isnull=True
            ).count()
            if orphaned_feedback > 0:
                issues.append(f"{orphaned_feedback} orphaned feedback entries")
            
            # Check for invalid star ratings
            invalid_ratings = PrioritizationFeedback.objects.filter(
                star_rating__lt=1
            ).count() + PrioritizationFeedback.objects.filter(
                star_rating__gt=5
            ).count()
            if invalid_ratings > 0:
                issues.append(f"{invalid_ratings} invalid star ratings")
            
            # Check for missing user analytics
            users_without_analytics = User.objects.filter(
                analytics__isnull=True
            ).count()
            if users_without_analytics > 0:
                issues.append(f"{users_without_analytics} users without analytics")
            
            # Check adjustment consistency
            inconsistent_adjustments = ScoringModelAdjustment.objects.filter(
                adjustment_delta__isnull=True
            ).count()
            if inconsistent_adjustments > 0:
                issues.append(f"{inconsistent_adjustments} adjustments with null deltas")
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Database Integrity",
                success=len(issues) == 0,
                message=f"{'No issues found' if len(issues) == 0 else f'{len(issues)} issues found'}",
                duration=duration,
                data={'issues': issues}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Database Integrity",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def reset_test_data(self) -> TestResult:
        """Clean up test data"""
        start_time = time.time()
        
        try:
            print("\n🧹 Cleaning up test data...")
            
            # Delete test feedback
            feedback_count = PrioritizationFeedback.objects.filter(
                user=self.test_user
            ).count()
            PrioritizationFeedback.objects.filter(user=self.test_user).delete()
            
            # Delete test adjustments
            adjustment_count = ScoringModelAdjustment.objects.filter(
                user=self.test_user
            ).count()
            ScoringModelAdjustment.objects.filter(user=self.test_user).delete()
            
            # Delete test study plan
            if self.test_plan:
                self.test_plan.delete()
            
            # Delete test user analytics
            UserAnalytics.objects.filter(user=self.test_user).delete()
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Data Cleanup",
                success=True,
                message=f"Cleaned {feedback_count} feedback, {adjustment_count} adjustments",
                duration=duration,
                data={
                    'feedback_deleted': feedback_count,
                    'adjustments_deleted': adjustment_count
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Data Cleanup",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def run_full_test_suite(self) -> List[TestResult]:
        """Run complete test suite"""
        print("🚀 Running full feedback system test suite...")
        print("=" * 60)
        
        self.test_results = []
        
        # Setup
        if not self.setup_test_environment():
            print("❌ Failed to setup test environment")
            return self.test_results
        
        # Run all tests
        tests = [
            self.test_feedback_collection,
            lambda: self.simulate_feedback_patterns(15),
            self.test_reinforcement_learning,
            self.validate_model_weights,
            self.benchmark_performance,
            self.check_database_integrity
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.test_results.append(result)
                self._print_test_result(result)
            except Exception as e:
                error_result = TestResult(
                    test_name=test_func.__name__,
                    success=False,
                    message=f"Unexpected error: {str(e)}",
                    duration=0.0
                )
                self.test_results.append(error_result)
                self._print_test_result(error_result)
        
        # Summary
        self._print_test_summary()
        
        return self.test_results
    
    def _print_test_result(self, result: TestResult):
        """Print individual test result"""
        status = "✅" if result.success else "❌"
        print(f"\n{status} {result.test_name}")
        print(f"   {result.message}")
        print(f"   Duration: {result.duration:.3f}s")
    
    def _print_test_summary(self):
        """Print test suite summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        total_duration = sum(r.duration for r in self.test_results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Duration: {total_duration:.3f}s")
        
        if total_tests - passed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"   - {result.test_name}: {result.message}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    debugger = FeedbackDebugger()
    
    if command == "test-collection":
        debugger.setup_test_environment()
        result = debugger.test_feedback_collection()
        debugger._print_test_result(result)
        
    elif command == "simulate-feedback":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        debugger.setup_test_environment()
        result = debugger.simulate_feedback_patterns(count)
        debugger._print_test_result(result)
        
    elif command == "test-rl":
        debugger.setup_test_environment()
        result = debugger.test_reinforcement_learning()
        debugger._print_test_result(result)
        
    elif command == "validate-weights":
        result = debugger.validate_model_weights()
        debugger._print_test_result(result)
        
    elif command == "benchmark":
        debugger.setup_test_environment()
        result = debugger.benchmark_performance()
        debugger._print_test_result(result)
        
    elif command == "check-db":
        result = debugger.check_database_integrity()
        debugger._print_test_result(result)
        
    elif command == "reset-test-data":
        result = debugger.reset_test_data()
        debugger._print_test_result(result)
        
    elif command == "full-test":
        debugger.run_full_test_suite()
        
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()

