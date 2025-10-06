from rest_framework import serializers
from .models import Upload, Plan

class UploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Upload
        fields = "__all__"
        read_only_fields = ["user", "status", "created_at"]

class PlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plan
        fields = "__all__"
