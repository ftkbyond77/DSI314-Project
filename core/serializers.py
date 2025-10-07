from rest_framework import serializers
from .models import Upload, Plan

class UploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Upload
        fields = ['id', 'filename', 'status', 'pages', 'created_at']
        read_only_fields = ['id', 'status', 'created_at']

class PlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plan
        fields = ['id', 'plan_json', 'score', 'created_at']
        read_only_fields = ['id', 'created_at']