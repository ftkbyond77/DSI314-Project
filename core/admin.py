from django.contrib import admin
from .models import Upload, Chunk, Plan

@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'filename', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('filename', 'user__username')


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ('id', 'upload', 'chunk_id', 'start_page', 'end_page', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('chunk_id', 'upload__filename')


@admin.register(Plan)
class PlanAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'upload', 'score', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('upload__filename', 'user__username')
