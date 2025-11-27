# core/admin.py

from django.contrib import admin
from .models import Upload, Chunk, StudyPlanHistory, Plan

@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    list_display = ('filename', 'user', 'status', 'ocr_used', 'ocr_pages', 'created_at')
    search_fields = ('filename', 'user__username')
    list_filter = ('status', 'ocr_used', 'created_at')
    readonly_fields = ('created_at',)

@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ('chunk_id', 'upload', 'start_page', 'end_page', 'embedding_id', 'created_at')
    search_fields = ('chunk_id', 'upload__filename')
    list_filter = ('created_at',)
    readonly_fields = ('created_at',)

@admin.register(StudyPlanHistory)
class StudyPlanHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at', 'total_files', 'total_pages', 'total_chunks', 'status')
    search_fields = ('user__username',)
    list_filter = ('status', 'sort_method', 'created_at')
    readonly_fields = ('created_at', 'updated_at',)
    filter_horizontal = ('uploads',)  

@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ('user', 'upload', 'created_at', 'score', 'version')
    search_fields = ('user__username', 'upload__filename')
    list_filter = ('version', 'created_at')
    readonly_fields = ('created_at',)

