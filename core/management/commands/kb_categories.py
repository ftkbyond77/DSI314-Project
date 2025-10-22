# core/management/commands/kb_categories.py - Category System Management

from django.core.management.base import BaseCommand
from core.knowledge_weighting import SchemaHandler, DynamicCategoryMapper
import json

class Command(BaseCommand):
    help = 'Manage dynamic category system'

    def add_arguments(self, parser):
        parser.add_argument(
            'action',
            type=str,
            choices=['list', 'discover', 'info', 'test', 'refresh', 'export', 'import'],
            help='Action to perform'
        )
        
        parser.add_argument(
            '--category',
            type=str,
            help='Category name for info/test actions'
        )
        
        parser.add_argument(
            '--file',
            type=str,
            help='File path for import/export'
        )
        
        parser.add_argument(
            '--input',
            type=str,
            help='Input string for test action'
        )

    def handle(self, *args, **options):
        action = options['action']
        
        self.stdout.write(self.style.SUCCESS(f'\n{"="*80}'))
        self.stdout.write(self.style.SUCCESS(f'Category System Manager - Action: {action}'))
        self.stdout.write(self.style.SUCCESS(f'{"="*80}\n'))
        
        if action == 'list':
            self.list_categories()
        
        elif action == 'discover':
            self.discover_categories()
        
        elif action == 'info':
            category = options.get('category')
            if not category:
                self.stdout.write(self.style.ERROR('Please provide --category'))
                return
            self.show_category_info(category)
        
        elif action == 'test':
            input_str = options.get('input')
            if not input_str:
                self.stdout.write(self.style.ERROR('Please provide --input'))
                return
            self.test_normalization(input_str)
        
        elif action == 'refresh':
            self.refresh_system()
        
        elif action == 'export':
            file_path = options.get('file', 'category_patterns.json')
            self.export_patterns(file_path)
        
        elif action == 'import':
            file_path = options.get('file')
            if not file_path:
                self.stdout.write(self.style.ERROR('Please provide --file'))
                return
            self.import_patterns(file_path)
    
    def list_categories(self):
        """List all discovered categories."""
        self.stdout.write('Discovered Categories:\n')
        
        categories = SchemaHandler.get_all_discovered_categories()
        
        if not categories:
            self.stdout.write(self.style.WARNING('No categories found'))
            return
        
        for idx, cat in enumerate(categories, 1):
            mapper = SchemaHandler.get_category_mapper()
            info = mapper.get_category_info(cat)
            
            learned_badge = '🆕' if info['is_learned'] else '📚'
            keyword_count = info['keyword_count']
            
            self.stdout.write(f"  {learned_badge} {idx:3d}. {cat:30} ({keyword_count} keywords)")
        
        self.stdout.write(f'\nTotal: {len(categories)} categories')
    
    def discover_categories(self):
        """Force category discovery from KB."""
        self.stdout.write('Discovering categories from KB...\n')
        
        mapper = SchemaHandler.get_category_mapper()
        mapper.refresh_patterns()
        
        categories = mapper.get_all_categories()
        
        self.stdout.write(self.style.SUCCESS(f'\n✅ Discovered {len(categories)} categories'))
        self.list_categories()
    
    def show_category_info(self, category: str):
        """Show detailed info about a category."""
        mapper = SchemaHandler.get_category_mapper()
        info = mapper.get_category_info(category)
        
        if not info.get('exists', True):
            self.stdout.write(self.style.WARNING(f'\nCategory "{category}" not found'))
            return
        
        self.stdout.write(f'\nCategory: {info["category"]}')
        self.stdout.write(f'Type: {"Learned from KB" if info["is_learned"] else "Seed pattern"}')
        self.stdout.write(f'Keyword count: {info["keyword_count"]}')
        
        if info['keywords']:
            self.stdout.write('\nKeywords:')
            for kw in sorted(info['keywords'])[:20]:
                self.stdout.write(f'  • {kw}')
            
            if len(info['keywords']) > 20:
                self.stdout.write(f'  ... and {len(info["keywords"]) - 20} more')
    
    def test_normalization(self, input_str: str):
        """Test category normalization."""
        self.stdout.write(f'Testing normalization for: "{input_str}"\n')
        
        mapper = SchemaHandler.get_category_mapper()
        
        # Test with learning disabled
        result_no_learn = mapper.normalize_category(input_str, learn=False)
        
        # Test with learning enabled
        result_with_learn = mapper.normalize_category(input_str, learn=True)
        
        self.stdout.write(f'Input:           "{input_str}"')
        self.stdout.write(f'Normalized:      "{result_no_learn}"')
        
        if result_no_learn != result_with_learn:
            self.stdout.write(f'Learned mapping: "{result_with_learn}"')
        
        # Show matched category info
        info = mapper.get_category_info(result_no_learn)
        if info.get('exists', True):
            self.stdout.write(f'\nMatched category has {info["keyword_count"]} keywords')
    
    def refresh_system(self):
        """Refresh entire category system."""
        self.stdout.write('Refreshing category system...\n')
        
        SchemaHandler.refresh_category_system()
        
        categories = SchemaHandler.get_all_discovered_categories()
        self.stdout.write(self.style.SUCCESS(f'\n✅ System refreshed'))
        self.stdout.write(f'Total categories: {len(categories)}')
    
    def export_patterns(self, file_path: str):
        """Export category patterns to file."""
        self.stdout.write(f'Exporting patterns to {file_path}...\n')
        
        mapper = SchemaHandler.get_category_mapper()
        patterns = mapper.export_patterns()
        
        with open(file_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(f'✅ Exported {len(patterns)} category patterns'))
    
    def import_patterns(self, file_path: str):
        """Import category patterns from file."""
        self.stdout.write(f'Importing patterns from {file_path}...\n')
        
        try:
            with open(file_path, 'r') as f:
                patterns = json.load(f)
            
            mapper = SchemaHandler.get_category_mapper()
            mapper.import_patterns(patterns)
            
            self.stdout.write(self.style.SUCCESS(f'✅ Imported {len(patterns)} category patterns'))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Import failed: {e}'))