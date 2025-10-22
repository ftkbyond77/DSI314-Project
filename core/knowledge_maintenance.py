# core/knowledge_maintenance.py - Knowledge Base Maintenance Tasks

from celery import shared_task
from django.core.cache import cache
from .knowledge_weighting import KnowledgeGroundingEngine, refresh_knowledge_base_cache
from .models import StudyPlanHistory
from datetime import datetime, timedelta
import json

@shared_task
def update_knowledge_base_stats():
    """
    Refresh knowledge base statistics after document ingestion.
    Run this task after uploading new document batches to the KB.
    
    Schedule: After batch uploads or daily at off-peak hours
    """
    print("🔄 Updating knowledge base statistics...")
    
    try:
        # Refresh cache
        stats = refresh_knowledge_base_cache()
        
        print(f"✅ KB statistics updated:")
        print(f"   Total categories: {stats['total_categories']}")
        print(f"   Most common: {stats['most_common_category']}")
        print(f"   Cache refreshed: {datetime.now()}")
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ KB stats update failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@shared_task
def analyze_kb_distribution():
    """
    Analyze knowledge base distribution and detect imbalances.
    Provides recommendations for rebalancing.
    
    Schedule: Weekly
    """
    print("📊 Analyzing KB distribution...")
    
    try:
        engine = KnowledgeGroundingEngine(namespace="knowledge_base")
        stats = engine.get_statistics_summary()
        
        category_dist = stats.get('category_distribution', {})
        
        if not category_dist:
            return {"success": False, "error": "No distribution data"}
        
        # Analyze balance
        total_docs = sum(category_dist.values())
        avg_per_category = total_docs / len(category_dist)
        
        imbalanced_categories = []
        for category, count in category_dist.items():
            ratio = count / avg_per_category
            if ratio < 0.3:
                imbalanced_categories.append({
                    "category": category,
                    "count": count,
                    "ratio": round(ratio, 2),
                    "status": "underrepresented"
                })
            elif ratio > 3.0:
                imbalanced_categories.append({
                    "category": category,
                    "count": count,
                    "ratio": round(ratio, 2),
                    "status": "overrepresented"
                })
        
        # Generate recommendations
        recommendations = []
        
        for item in imbalanced_categories:
            if item['status'] == 'underrepresented':
                recommendations.append(
                    f"Add more {item['category']} materials to improve coverage "
                    f"(currently only {item['ratio']*100:.0f}% of average)"
                )
            else:
                recommendations.append(
                    f"{item['category']} is heavily represented ({item['ratio']*100:.0f}% of average). "
                    f"Consider diversity in other domains."
                )
        
        print(f"✅ Distribution analysis complete")
        print(f"   Total documents: {total_docs}")
        print(f"   Imbalanced categories: {len(imbalanced_categories)}")
        
        return {
            "success": True,
            "total_documents": total_docs,
            "categories": len(category_dist),
            "avg_per_category": round(avg_per_category, 0),
            "imbalanced": imbalanced_categories,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Distribution analysis failed: {e}")
        return {"success": False, "error": str(e)}


@shared_task
def validate_kb_grounding_quality():
    """
    Validate quality of KB grounding in recent plans.
    Detects issues with confidence, coverage, etc.
    
    Schedule: Daily
    """
    print("🔍 Validating KB grounding quality...")
    
    try:
        # Get recent plans (last 7 days)
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
        
        for plan in recent_plans:
            plan_data = plan.plan_json
            
            for item in plan_data:
                if item.get('file') == '📅 WEEKLY SCHEDULE':
                    continue
                
                total_tasks += 1
                
                kb_confidence = item.get('kb_confidence', 0)
                kb_relevance = item.get('kb_relevance', 0)
                kb_depth = item.get('kb_depth', 'unknown')
                
                if kb_confidence > 0:
                    kb_enabled_tasks += 1
                    confidence_scores.append(kb_confidence)
                    relevance_scores.append(kb_relevance)
                    
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
        
        # Quality assessment
        issues = []
        
        if kb_usage_rate < 50:
            issues.append(f"Low KB usage rate ({kb_usage_rate:.0f}%). Check if KB grounding is enabled.")
        
        if avg_confidence < 0.4:
            issues.append(f"Low average confidence ({avg_confidence:.2f}). KB may need more diverse content.")
        
        if low_confidence_rate > 40:
            issues.append(f"High rate of low-confidence tasks ({low_confidence_rate:.0f}%). Consider KB expansion.")
        
        if no_kb_coverage > total_tasks * 0.3:
            issues.append(f"Many tasks have minimal KB coverage ({no_kb_coverage}/{total_tasks}). Add more reference materials.")
        
        status = "healthy" if not issues else "needs_attention"
        
        print(f"✅ Quality validation complete")
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
            "no_kb_coverage_tasks": no_kb_coverage,
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Quality validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@shared_task
def clear_kb_cache():
    """
    Clear all KB-related cache.
    Use when KB has been significantly updated.
    """
    print("🗑️ Clearing KB cache...")
    
    try:
        cache_keys = [
            "kb_category_stats_v2",
            "kb_global_stats_v2",
            "kb_source_distribution_v2"
        ]
        
        for key in cache_keys:
            cache.delete(key)
        
        print("✅ KB cache cleared")
        return {"success": True, "cleared_keys": cache_keys}
        
    except Exception as e:
        print(f"❌ Cache clear failed: {e}")
        return {"success": False, "error": str(e)}


@shared_task
def generate_kb_health_report():
    """
    Generate comprehensive KB health report.
    
    Schedule: Weekly
    """
    print("📋 Generating KB health report...")
    
    try:
        # Run all checks
        stats_result = update_knowledge_base_stats.apply().get()
        dist_result = analyze_kb_distribution.apply().get()
        quality_result = validate_kb_grounding_quality.apply().get()
        
        # Compile report
        report = {
            "generated_at": datetime.now().isoformat(),
            "overall_status": "healthy",
            "statistics": stats_result,
            "distribution": dist_result,
            "quality": quality_result,
            "summary": [],
            "action_items": []
        }
        
        # Determine overall status
        if quality_result.get('status') == 'needs_attention':
            report['overall_status'] = 'needs_attention'
        
        if dist_result.get('imbalanced'):
            report['overall_status'] = 'needs_attention'
        
        # Generate summary
        report['summary'].append(
            f"KB contains {stats_result.get('stats', {}).get('total_categories', 0)} categories"
        )
        
        if quality_result.get('kb_usage_rate'):
            report['summary'].append(
                f"KB grounding used in {quality_result['kb_usage_rate']:.0f}% of recent tasks"
            )
        
        report['summary'].append(
            f"Average confidence: {quality_result.get('avg_confidence', 0):.2f}"
        )
        
        # Action items
        if quality_result.get('issues'):
            report['action_items'].extend(quality_result['issues'])
        
        if dist_result.get('recommendations'):
            report['action_items'].extend(dist_result['recommendations'])
        
        print(f"✅ Health report generated")
        print(f"   Status: {report['overall_status']}")
        print(f"   Action items: {len(report['action_items'])}")
        
        # Save report (optional - can save to database)
        # KBHealthReport.objects.create(report=json.dumps(report))
        
        return report
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# ==================== MANUAL TRIGGERS ====================

def manual_kb_refresh():
    """
    Manually trigger KB cache refresh.
    Call this from Django shell after batch uploads.
    """
    print("🔄 Manual KB refresh initiated...")
    result = update_knowledge_base_stats.apply()
    return result.get()


def manual_kb_health_check():
    """
    Manually trigger KB health check.
    Call this from Django shell for diagnostics.
    """
    print("🏥 Manual KB health check initiated...")
    result = generate_kb_health_report.apply()
    return result.get()