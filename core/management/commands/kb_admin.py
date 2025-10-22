# core/management/commands/kb_admin.py - KB Administration Commands

from django.core.management.base import BaseCommand
from core.knowledge_weighting import (
    test_knowledge_grounding,
    validate_schema_handling,
    refresh_knowledge_base_cache,
    KnowledgeGroundingEngine
)
from core.knowledge_maintenance import (
    manual_kb_refresh,
    manual_kb_health_check
)
import json

class Command(BaseCommand):
    help = 'Knowledge Base administration commands'

    def add_arguments(self, parser):
        parser.add_argument(
            'action',
            type=str,
            help='Action to perform: test, validate, refresh, health, stats'
        )
        
        parser.add_argument(
            '--namespace',
            type=str,
            default='knowledge_base',
            help='Pinecone namespace'
        )

    def handle(self, *args, **options):
        action = options['action']
        namespace = options['namespace']
        
        self.stdout.write(self.style.SUCCESS(f'\n{"="*80}'))
        self.stdout.write(self.style.SUCCESS(f'Knowledge Base Admin - Action: {action}'))
        self.stdout.write(self.style.SUCCESS(f'{"="*80}\n'))
        
        if action == 'test':
            self.test_kb_grounding()
        
        elif action == 'validate':
            self.validate_schema()
        
        elif action == 'refresh':
            self.refresh_cache()
        
        elif action == 'health':
            self.health_check()
        
        elif action == 'stats':
            self.show_stats(namespace)
        
        else:
            self.stdout.write(self.style.ERROR(f'Unknown action: {action}'))
            self.stdout.write('Valid actions: test, validate, refresh, health, stats')
    
    def test_kb_grounding(self):
        """Test KB grounding system"""
        self.stdout.write('Running KB grounding tests...\n')
        
        try:
            result, enhanced = test_knowledge_grounding()
            
            self.stdout.write(self.style.SUCCESS('✅ Test passed'))
            self.stdout.write(f'\nKB Relevance Score: {result["knowledge_relevance_score"]:.3f}')
            self.stdout.write(f'Confidence: {result["confidence"]:.3f}')
            self.stdout.write(f'Knowledge Depth: {result["knowledge_depth"]}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Test failed: {e}'))
    
    def validate_schema(self):
        """Validate schema handling"""
        self.stdout.write('Validating schema flexibility...\n')
        
        try:
            validate_schema_handling()
            self.stdout.write(self.style.SUCCESS('✅ Schema validation passed'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Validation failed: {e}'))
    
    def refresh_cache(self):
        """Refresh KB cache"""
        self.stdout.write('Refreshing KB statistics cache...\n')
        
        try:
            stats = refresh_knowledge_base_cache()
            
            self.stdout.write(self.style.SUCCESS('✅ Cache refreshed'))
            self.stdout.write(f'\nStatistics:')
            self.stdout.write(f'  Total categories: {stats["total_categories"]}')
            self.stdout.write(f'  Most common: {stats["most_common_category"]}')
            self.stdout.write(f'  Least common: {stats["least_common_category"]}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Refresh failed: {e}'))
    
    def health_check(self):
        """Run health check"""
        self.stdout.write('Running KB health check...\n')
        
        try:
            report = manual_kb_health_check()
            
            status = report.get('overall_status', 'unknown')
            style = self.style.SUCCESS if status == 'healthy' else self.style.WARNING
            
            self.stdout.write(style(f'\nOverall Status: {status.upper()}'))
            
            if report.get('summary'):
                self.stdout.write('\nSummary:')
                for item in report['summary']:
                    self.stdout.write(f'  • {item}')
            
            if report.get('action_items'):
                self.stdout.write(self.style.WARNING('\nAction Items:'))
                for item in report['action_items']:
                    self.stdout.write(self.style.WARNING(f'  ⚠ {item}'))
            else:
                self.stdout.write(self.style.SUCCESS('\n✅ No action items'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Health check failed: {e}'))
    
    def show_stats(self, namespace):
        """Show KB statistics"""
        self.stdout.write(f'Fetching KB statistics (namespace: {namespace})...\n')
        
        try:
            engine = KnowledgeGroundingEngine(namespace=namespace)
            stats = engine.get_statistics_summary()
            
            self.stdout.write(self.style.SUCCESS('✅ Statistics retrieved\n'))
            
            self.stdout.write('Category Distribution:')
            cat_dist = stats.get('category_distribution', {})
            for category, count in sorted(cat_dist.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(count / max(cat_dist.values()) * 50)
                self.stdout.write(f'  {category:20} {count:5} {bar}')
            
            self.stdout.write('\nGlobal Statistics:')
            global_stats = stats.get('global_similarity_stats', {})
            for key, value in global_stats.items():
                self.stdout.write(f'  {key:15} {value:.3f}' if isinstance(value, float) else f'  {key:15} {value}')
            
            self.stdout.write(f'\nNamespace: {stats.get("namespace", "unknown")}')
            self.stdout.write(f'Cache enabled: {stats.get("cache_enabled", False)}')
            self.stdout.write(f'Timestamp: {stats.get("timestamp", "unknown")}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Failed to fetch stats: {e}'))