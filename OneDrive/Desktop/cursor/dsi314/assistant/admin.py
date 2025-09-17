from django.contrib import admin
from .models import UploadedDocument, ProcessingRun


@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "original_name", "created_at")
    search_fields = ("original_name",)


@admin.register(ProcessingRun)
class ProcessingRunAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at")
    search_fields = ("id",)
    autocomplete_fields = ("documents",)

