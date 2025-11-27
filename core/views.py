# core/views.py

import csv
from django.http import HttpResponse
from django.shortcuts import render
from django.apps import apps
from django.contrib.auth.decorators import user_passes_test, login_required

def is_admin(user):
    return user.is_authenticated and user.is_staff

@login_required
@user_passes_test(is_admin)
def admin_csv_page(request):
    """
    Renders a page listing all models available for export.
    """
    # List of models from core/models.py
    models_list = [
        'Upload', 
        'Chunk', 
        'StudyPlanHistory', 
        'Plan', 
        'QuizSession', 
        'QuizQuestion', 
        'QuizAnswer', 
        'PrioritizationFeedback', 
        'ScoringModelAdjustment', 
        'UserAnalytics'
    ]
    
    return render(request, 'core/admin_csv.html', {'models_list': models_list})

@login_required
@user_passes_test(is_admin)
def export_model_csv(request, model_name):
    """
    Generates a CSV response for the specified model.
    """
    try:
        # Dynamically get the model from the 'core' app
        model = apps.get_model('core', model_name)
    except LookupError:
        return HttpResponse(f"Model '{model_name}' not found.", status=404)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{model_name}_export.csv"'

    writer = csv.writer(response)
    
    # Get all field names from the model
    fields = [field.name for field in model._meta.fields]
    
    # Write the header row
    writer.writerow(fields)

    # Write data rows
    queryset = model.objects.all().order_by('-id')
    
    for obj in queryset:
        row = []
        for field in fields:
            value = getattr(obj, field)
            if value is None:
                value = ''
            row.append(str(value))
        writer.writerow(row)

    return response