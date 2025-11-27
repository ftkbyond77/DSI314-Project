# core/knowledge_maintenance.py - Maintenance Tasks for Dynamic System

from celery import shared_task
from django.core.cache import cache
from .knowledge_weighting import (
    refresh_knowledge_base_cache,
    KnowledgeGroundingEngine,
    SchemaHandler,
    DynamicCategoryMapper,
    KnowledgeStatisticsEngine
)
from .models import StudyPlanHistory
from datetime import datetime, timedelta
import json
import re

@shared_task
def update_knowledge_base_stats():
    """
    Refresh all KB statistics and caches.
    Run after batch document uploads.
    """
    print("Updating knowledge base statistics...")
    
    try:
        stats = refresh_knowledge_base_cache()
        
        print(f"   KB statistics updated:")
        print(f"   Categories: {stats['category_system']['total_categories']}")
        print(f"   Tracked stats: {stats['statistics']['categories_tracked']}")
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"KB stats update failed: {e}")
        return {"success": False, "error": str(e)}


@shared_task
def discover_new_categories():
    """
    Force category discovery from KB.
    Run weekly to find new categories.
    """
    print("Discovering new categories...")
    
    try:
        mapper = SchemaHandler.get_category_mapper()
        
        # Refresh patterns (triggers KB sampling)
        mapper.refresh_patterns()
        
        categories = mapper.get_all_categories()
        
        print(f"   Discovery complete:")
        print(f"   Total categories: {len(categories)}")
        
        return {
            "success": True,
            "total_categories": len(categories),
            "categories": categories,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Category discovery failed: {e}")
        return {"success": False, "error": str(e)}


@shared_task
def analyze_kb_distribution():
    """
    Analyze KB distribution and detect imbalances.
    Uses dynamic statistics engine.
    """
    print("Analyzing KB distribution...")
    
    try:
        stats_engine = KnowledgeStatisticsEngine(namespace="knowledge_base")
        category_stats = stats_engine.get_category_statistics(force_refresh=True)
        
        if not category_stats:
            return {"success": False, "error": "No distribution data"}
        
        # Calculate balance metrics
        doc_counts = [info.get('document_count', 0) for info in category_stats.values()]
        total_docs = sum(doc_counts)
        avg_per_category = total_docs / len(category_stats) if category_stats else 0
        
        imbalanced_categories = []
        well_balanced_categories = []
        
        for category, cat_info in category_stats.items():
            count = cat_info.get('document_count', 0)
            if avg_per_category > 0:
                ratio = count / avg_per_category
                
                if ratio < 0.3:
                    imbalanced_categories.append({
                        "category": category,
                        "count": count,
                        "ratio": round(ratio, 2),
                        "status": "underrepresented",
                        "avg_similarity": cat_info.get('avg_similarity', 0)
                    })
                elif ratio > 3.0:
                    imbalanced_categories.append({
                        "category": category,
                        "count": count,
                        "ratio": round(ratio, 2),
                        "status": "overrepresented",
                        "avg_similarity": cat_info.get('avg_similarity', 0)
                    })
                else:
                    well_balanced_categories.append(category)
        
        # Generate recommendations
        recommendations = []
        
        for item in imbalanced_categories:
            if item['status'] == 'underrepresented':
                recommendations.append({
                    "priority": "high" if item['ratio'] < 0.1 else "medium",
                    "action": f"Add more {item['category']} materials",
                    "reason": f"Currently only {item['ratio']*100:.0f}% of average coverage",
                    "current_count": item['count']
                })
            else:
                recommendations.append({
                    "priority": "low",
                    "action": f"Diversify beyond {item['category']}",
                    "reason": f"Over-represented at {item['ratio']*100:.0f}% of average",
                    "current_count": item['count']
                })
        
        print(f"   Distribution analysis complete")
        print(f"   Total documents: {total_docs}")
        print(f"   Imbalanced categories: {len(imbalanced_categories)}")
        print(f"   Well-balanced categories: {len(well_balanced_categories)}")
        
        return {
            "success": True,
            "total_documents": total_docs,
            "total_categories": len(category_stats),
            "avg_per_category": round(avg_per_category, 0),
            "well_balanced": well_balanced_categories,
            "imbalanced": imbalanced_categories,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Distribution analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@shared_task
def validate_kb_grounding_quality():
    """
    Validate quality of KB grounding in recent plans.
    """
    print("Validating KB grounding quality...")
    
    try:
        # Get recent plans
        cutoff = datetime.now() - timedelta(days=7)
        recent_plans = StudyPlanHistory.objects.filter(
            created_at__gte=cutoff
        ).order_by('-created_at')[:100]
        
        if not recent_plans:
            return {"success": False, "message": "No recent plans to analyze"}
        
        # Analyze KB metrics
        total_tasks = 0
        kb_enabled_tasks = 0
        low_confidence_tasks = 0
        high_confidence_tasks = 0
        no_kb_coverage = 0
        
        confidence_scores = []
        relevance_scores = []
        depth_distribution = defaultdict(int)
        
        for plan in recent_plans:
            plan_data = plan.plan_json
            
            for item in plan_data:
                if item.get('file') == 'WEEKLY SCHEDULE':
                    continue
                
                total_tasks += 1
                
                kb_confidence = item.get('kb_confidence', 0)
                kb_relevance = item.get('kb_relevance', 0)
                kb_depth = item.get('kb_depth', 'unknown')
                
                if kb_confidence > 0:
                    kb_enabled_tasks += 1
                    confidence_scores.append(kb_confidence)
                    relevance_scores.append(kb_relevance)
                    depth_distribution[kb_depth] += 1
                    
                    if kb_confidence < 0.3:
                        low_confidence_tasks += 1
                    elif kb_confidence > 0.7:
                        high_confidence_tasks += 1
                    
                    if kb_depth in ['none', 'minimal']:
                        no_kb_coverage += 1
        
        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        kb_usage_rate = (kb_enabled_tasks / total_tasks * 100) if total_tasks > 0 else 0
        low_confidence_rate = (low_confidence_tasks / kb_enabled_tasks * 100) if kb_enabled_tasks > 0 else 0
        high_confidence_rate = (high_confidence_tasks / kb_enabled_tasks * 100) if kb_enabled_tasks > 0 else 0
        
        # Quality assessment
        issues = []
        successes = []
        
        if kb_usage_rate < 50:
            issues.append(f"Low KB usage rate ({kb_usage_rate:.0f}%). Check if KB grounding is enabled.")
        else:
            successes.append(f"Good KB usage rate ({kb_usage_rate:.0f}%)")
        
        if avg_confidence < 0.4:
            issues.append(f"Low average confidence ({avg_confidence:.2f}). KB may need more diverse content.")
        elif avg_confidence > 0.6:
            successes.append(f"High average confidence ({avg_confidence:.2f})")
        
        if low_confidence_rate > 40:
            issues.append(f"High rate of low-confidence tasks ({low_confidence_rate:.0f}%). Consider KB expansion.")
        
        if high_confidence_rate > 40:
            successes.append(f"High rate of high-confidence tasks ({high_confidence_rate:.0f}%)")
        
        if no_kb_coverage > total_tasks * 0.3:
            issues.append(f"Many tasks have minimal KB coverage ({no_kb_coverage}/{total_tasks}). Add more reference materials.")
        
        status = "healthy" if not issues or len(successes) > len(issues) else "needs_attention"
        
        print(f"   Quality validation complete")
        print(f"   Status: {status}")
        print(f"   KB usage: {kb_usage_rate:.1f}%")
        print(f"   Avg confidence: {avg_confidence:.3f}")
        
        return {
            "success": True,
            "status": status,
            "total_tasks_analyzed": total_tasks,
            "kb_enabled_tasks": kb_enabled_tasks,
            "kb_usage_rate": round(kb_usage_rate, 1),
            "avg_confidence": round(avg_confidence, 3),
            "avg_relevance": round(avg_relevance, 3),
            "low_confidence_tasks": low_confidence_tasks,
            "high_confidence_tasks": high_confidence_tasks,
            "low_confidence_rate": round(low_confidence_rate, 1),
            "high_confidence_rate": round(high_confidence_rate, 1),
            "no_kb_coverage_tasks": no_kb_coverage,
            "depth_distribution": dict(depth_distribution),
            "issues": issues,
            "successes": successes,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Quality validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@shared_task
def calibrate_thresholds():
    """
    Recalibrate adaptive thresholds based on current KB data.
    Run weekly for optimal threshold adjustment.
    """
    print("Calibrating adaptive thresholds...")
    
    try:
        from .knowledge_weighting import AdaptiveThresholdEngine
        
        threshold_engine = AdaptiveThresholdEngine()
        
        # Clear cache to force recomputation
        threshold_engine.clear_cache()
        
        # Get new thresholds
        thresholds = threshold_engine._get_adaptive_thresholds()
        
        global_thresholds = thresholds.get('_global', {})
        
        print(f"   Thresholds calibrated:")
        print(f"   Min similarity: {global_thresholds.get('min_similarity', 0):.3f}")
        print(f"   High similarity: {global_thresholds.get('high_similarity', 0):.3f}")
        print(f"   High confidence: {global_thresholds.get('high_confidence', 0):.3f}")
        
        return {
            "success": True,
            "global_thresholds": global_thresholds,
            "category_count": len(thresholds) - 1,  # Exclude _global
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Threshold calibration failed: {e}")
        return {"success": False, "error": str(e)}


@shared_task
def update_calibration_parameters():
    """
    Update dynamic calibration parameters.
    Run weekly for optimal calibration.
    """
    print("Updating calibration parameters...")
    
    try:
        from .knowledge_weighting import DynamicCalibrationEngine
        
        calibration_engine = DynamicCalibrationEngine()
        
        # Clear cache to force recomputation
        calibration_engine.clear_cache()
        
        # Force recomputation
        params = calibration_engine._compute_all_calibration_parameters()
        
        # Get summary statistics
        total_categories = len(params) - 1  # Exclude _default
        avg_boost = sum(p.get('boost', 1.0) for p in params.values()) / len(params) if params else 1.0
        
        print(f"  Calibration parameters updated:")
        print(f"   Categories calibrated: {total_categories}")
        print(f"   Average boost factor: {avg_boost:.3f}")
        
        return {
            "success": True,
            "categories_calibrated": total_categories,
            "avg_boost": round(avg_boost, 3),
            "sample_params": {
                k: v for k, v in list(params.items())[:3]  # Show first 3
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f" Calibration update failed: {e}")
        return {"success": False, "error": str(e)}


@shared_task
def generate_kb_health_report():
    """
    Generate comprehensive KB health report.
    """
    print("Generating KB health report...")
    
    try:
        # Run all checks
        stats_result = update_knowledge_base_stats.apply().get()
        category_result = discover_new_categories.apply().get()
        dist_result = analyze_kb_distribution.apply().get()
        quality_result = validate_kb_grounding_quality.apply().get()
        threshold_result = calibrate_thresholds.apply().get()
        calibration_result = update_calibration_parameters.apply().get()
        
        # Compile report
        report = {
            "generated_at": datetime.now().isoformat(),
            "overall_status": "healthy",
            "statistics": stats_result,
            "categories": category_result,
            "distribution": dist_result,
            "quality": quality_result,
            "thresholds": threshold_result,
            "calibration": calibration_result,
            "summary": [],
            "action_items": [],
            "health_score": 0
        }
        
        # Calculate health score (0-100)
        health_score = 100
        
        # Deduct for issues
        if quality_result.get('status') == 'needs_attention':
            health_score -= 20
        
        if dist_result.get('imbalanced'):
            health_score -= 10 * len(dist_result['imbalanced'])
        
        if quality_result.get('kb_usage_rate', 0) < 50:
            health_score -= 15
        
        if quality_result.get('avg_confidence', 0) < 0.4:
            health_score -= 15
        
        health_score = max(0, min(100, health_score))
        report['health_score'] = health_score
        
        # Determine overall status
        if health_score >= 80:
            report['overall_status'] = 'excellent'
        elif health_score >= 60:
            report['overall_status'] = 'good'
        elif health_score >= 40:
            report['overall_status'] = 'needs_attention'
        else:
            report['overall_status'] = 'critical'
        
        # Generate summary
        report['summary'].append(
            f"KB Health Score: {health_score}/100 ({report['overall_status'].upper()})"
        )
        
        if category_result.get('total_categories'):
            report['summary'].append(
                f"Total categories: {category_result['total_categories']} (dynamic discovery active)"
            )
        
        if quality_result.get('kb_usage_rate'):
            report['summary'].append(
                f"KB grounding usage: {quality_result['kb_usage_rate']:.0f}%"
            )
        
        if quality_result.get('avg_confidence'):
            report['summary'].append(
                f"Average confidence: {quality_result['avg_confidence']:.2f}"
            )
        
        if threshold_result.get('success'):
            report['summary'].append(
                "Adaptive thresholds: calibrated and operational"
            )
        
        if calibration_result.get('success'):
            report['summary'].append(
                f"Dynamic calibration: {calibration_result['categories_calibrated']} categories"
            )
        
        # Collect action items
        if quality_result.get('issues'):
            for issue in quality_result['issues']:
                report['action_items'].append({
                    "priority": "high",
                    "category": "quality",
                    "issue": issue
                })
        
        if dist_result.get('recommendations'):
            for rec in dist_result['recommendations']:
                report['action_items'].append({
                    "priority": rec.get('priority', 'medium'),
                    "category": "distribution",
                    "issue": rec['action'],
                    "reason": rec['reason']
                })
        
        # Sort action items by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        report['action_items'].sort(
            key=lambda x: priority_order.get(x.get('priority', 'medium'), 1)
        )
        
        print(f"   Health report generated")
        print(f"   Health Score: {health_score}/100")
        print(f"   Status: {report['overall_status'].upper()}")
        print(f"   Action items: {len(report['action_items'])}")
        
        # Optionally save to database
        # KBHealthReport.objects.create(
        #     report=json.dumps(report),
        #     health_score=health_score,
        #     status=report['overall_status']
        # )
        
        return report
        
    except Exception as e:
        print(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@shared_task
def clear_all_kb_caches():
    """
    Clear all KB-related caches.
    Use when KB has been significantly updated.
    """
    print("Clearing all KB caches...")
    
    try:
        # Clear Django caches
        cache_keys = [
            "kb_dynamic_calibration_v1",
            "kb_adaptive_thresholds_v1",
            "kb_category_stats_v3",
            "kb_similarity_stats_v3",
            "kb_dynamic_categories_v1",
            "kb_category_patterns_v1"
        ]
        
        cleared_count = 0
        for key in cache_keys:
            try:
                cache.delete(key)
                cleared_count += 1
            except Exception as e:
                print(f"⚠️ Failed to clear {key}: {e}")
        
        print(f"Cleared {cleared_count}/{len(cache_keys)} cache keys")
        
        return {
            "success": True,
            "cleared_keys": cleared_count,
            "total_keys": len(cache_keys)
        }
        
    except Exception as e:
        print(f"Cache clear failed: {e}")
        return {"success": False, "error": str(e)}


# ==================== MANUAL TRIGGERS ====================

def manual_kb_refresh():
    """
    Manually trigger comprehensive KB refresh.
    """
    print("Manual comprehensive KB refresh initiated...")
    
    # Clear all caches first
    clear_all_kb_caches.apply()
    
    # Update all systems
    result = update_knowledge_base_stats.apply()
    return result.get()


def manual_kb_health_check():
    """
    Manually trigger KB health check.
    """
    print("Manual KB health check initiated...")
    result = generate_kb_health_report.apply()
    return result.get()


def manual_category_discovery():
    """
    Manually trigger category discovery.
    """
    print("Manual category discovery initiated...")
    result = discover_new_categories.apply()
    return result.get()