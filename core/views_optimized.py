# core/views_optimized.py - Production-Grade Views with AI Batch Processing

from django.contrib.auth import authenticate, login, logout
from .forms import LoginForm, RegistrationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import Upload, Plan, Chunk, StudyPlanHistory, QuizSession
from .tasks import process_upload
from .tasks_agentic_optimized import generate_optimized_plan_async
from .agent_tools_advanced import ToolLogger
from celery.result import AsyncResult
from typing import List, Dict
import json
import time

User = get_user_model()

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_WAIT_TIME = 300

def safe_int(value, default=0):
    """Safely convert to int"""
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def _bucket_tasks_kanban(tasks):
    """Helper to bucket tasks into High Quality, Back Log, and Validated"""
    columns = {
        'high_quality': [],
        'back_log': [],
        'validated': []  # for user validation
    }
    
    # Sort by priority first
    tasks_sorted = sorted(tasks, key=lambda x: x.get('priority', 999))
    
    for task in tasks_sorted:
        priority = task.get('priority', 999)
        urgency = task.get('urgency', 0)
        
        # Logic: Priorities 1-5 OR Very High Urgency goes to High Quality
        # Everything else goes to Back Log
        if priority <= 5 or urgency >= 8:
            columns['high_quality'].append(task)
        else:
            columns['back_log'].append(task)
            
    return columns

@login_required
def upload_page_optimized(request):
    """Optimized upload with AI batch processing support"""
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        
        user_goal = request.POST.get('user_goal', 'finish semester with good grades').strip()
        sort_method = request.POST.get('sort_method', 'hybrid')
        constraint_prompt = request.POST.get('constraint_prompt', '').strip()
        
        time_input = {
            'years': safe_int(request.POST.get('years')),
            'months': safe_int(request.POST.get('months')),
            'weeks': safe_int(request.POST.get('weeks')),
            'days': safe_int(request.POST.get('days')),
            'hours': safe_int(request.POST.get('hours'))
        }
        
        if not files:
            messages.error(request, 'Please select at least one PDF file.')
            return render(request, 'core/upload.html', {})
        
        if len(files) > 10:
            messages.error(request, 'Maximum 10 files allowed.')
            return render(request, 'core/upload.html', {})
        
        oversized_files = []
        for file in files:
            if not file.name.lower().endswith('.pdf'):
                messages.error(request, f'Only PDF files allowed: {file.name}')
                return render(request, 'core/upload.html', {})
            
            if file.size > MAX_FILE_SIZE:
                oversized_files.append((file.name, file.size / (1024 * 1024)))
        
        if oversized_files:
            for name, size_mb in oversized_files:
                messages.error(request, f'File too large: {name} ({size_mb:.1f}MB). Max 100MB.')
            return render(request, 'core/upload.html', {})
        
        upload_ids = []
        new_uploads = []
        total_size = 0
        
        for file in files:
            file_size_mb = file.size / (1024 * 1024)
            total_size += file_size_mb
            
            existing = Upload.objects.filter(
                user=request.user,
                filename=file.name,
                status='processed',
                pages__isnull=False
            ).first()
            
            if existing:
                print(f"Cache hit: {file.name}")
                upload_ids.append(existing.id)
            else:
                upload = Upload.objects.create(
                    user=request.user,
                    file=file,
                    filename=file.name,
                    status='uploaded'
                )
                new_uploads.append(upload)
                upload_ids.append(upload.id)
                print(f"New upload: {file.name} ({file_size_mb:.1f}MB)")
        
        for upload in new_uploads:
            process_upload.delay(upload.id)
        
        if total_size > 50:
            messages.info(
                request,
                f'Processing {len(new_uploads)} large files ({total_size:.1f}MB). Running in background.'
            )
        
        if new_uploads:
            start = time.time()
            quick_timeout = min(30, MAX_WAIT_TIME)
            
            while time.time() - start < quick_timeout:
                all_done = all(
                    Upload.objects.get(id=u.id).status in ['processed', 'failed']
                    for u in new_uploads
                )
                
                if all_done:
                    print(f"Fast processing: {time.time() - start:.1f}s")
                    break
                
                time.sleep(1)
            
            for u in new_uploads:
                u.refresh_from_db()
                if u.status == 'failed':
                    messages.error(request, f"Failed to process: {u.filename}")
                elif u.status != 'processed':
                    messages.warning(request, f"Still processing: {u.filename}")
        
        print(f"Launching agentic planning (batch mode, AI extraction)...")
        
        task = generate_optimized_plan_async.delay(
            user_id=request.user.id,
            upload_ids=upload_ids,
            user_goal=user_goal,
            time_input=time_input,
            constraints=constraint_prompt,
            sort_method=sort_method
        )
        
        print(f"Task ID: {task.id}")
        
        request.session['planning_task_id'] = task.id
        request.session['planning_params'] = {
            'user_goal': user_goal,
            'time_input': time_input,
            'constraints': constraint_prompt,
            'sort_method': sort_method,
            'file_count': len(files)
        }
        
        messages.success(
            request,
            f'AI analysis started for {len(files)} files using {sort_method} method.'
        )
        
        return redirect('planning_progress')
    
    user_uploads = Upload.objects.filter(
        user=request.user,
        status='processed'
    ).order_by('-created_at')[:10]
    
    history_plans = StudyPlanHistory.objects.filter(
        user=request.user,
        status='active'
    ).order_by('-created_at')[:10]
    
    quiz_history = QuizSession.objects.filter(
        user=request.user
    ).order_by('-created_at')[:20]
    
    return render(request, 'core/upload.html', {
        'recent_uploads': user_uploads,
        'history_plans': history_plans,
        'quiz_history': quiz_history 
    })


@login_required
def planning_progress(request):
    """Progress tracking"""
    task_id = request.session.get('planning_task_id')
    params = request.session.get('planning_params', {})
    
    if not task_id:
        messages.error(request, 'No planning task found')
        return redirect('upload_page_optimized')
    
    return render(request, 'core/planning_progress.html', {
        'task_id': task_id,
        'params': params
    })


@login_required
@require_http_methods(["GET"])
def planning_status_api(request, task_id):
    """Status API"""
    result = AsyncResult(task_id)
    
    response = {
        "status": result.state,
        "ready": result.ready(),
        "task_id": task_id
    }
    
    if result.state == 'PROCESSING':
        response["meta"] = result.info
        response["message"] = result.info.get('status', 'Processing...')
        response["progress"] = {
            "current": result.info.get('current', 0),
            "total": result.info.get('total', 3)
        }
    elif result.ready():
        if result.successful():
            data = result.result
            if data.get("success"):
                response["success"] = True
                response["plan_id"] = data.get("plan_id")
                response["history_id"] = data.get("history_id")
                response["execution_time"] = data.get("execution_time")
                response["tool_summary"] = data.get("tool_summary")
            else:
                response["error"] = data.get("error", "Unknown error")
                response["error_details"] = data.get("error_trace", "")[:500]
        else:
            response["error"] = str(result.info)
    elif result.state == 'PENDING':
        response["message"] = "Task queued..."
    elif result.state == 'FAILURE':
        response["error"] = str(result.info)
    
    return JsonResponse(response)


@login_required
def result_page_optimized(request):
    """Results with AI metrics"""
    latest_plan = Plan.objects.filter(
        user=request.user
    ).order_by('-created_at').first()
    
    if latest_plan:
        plan = latest_plan.plan_json
        
        uploads = Upload.objects.filter(
            user=request.user,
            status='processed'
        ).order_by('-created_at')[:10]
        
        schedule = None
        tasks = []
        metadata = {}
        
        for item in plan:
            if item.get('file') == '📅 WEEKLY SCHEDULE' or item.get('file') == 'WEEKLY SCHEDULE':
                schedule = item.get('schedule', [])
                metadata = item.get('metadata', {})
            else:
                tasks.append(item)
        
        # --- NEW KANBAN LOGIC ---
        kanban_columns = _bucket_tasks_kanban(tasks)
        # ------------------------
        
        total_files = len(tasks)
        total_chunks = sum(p.get('chunk_count', 0) for p in tasks)
        total_pages = sum(p.get('pages', 0) for p in tasks)
        total_hours = sum(p.get('estimated_hours', 0) for p in tasks)
        
        ocr_pages_total = sum(
            u.ocr_pages for u in uploads if hasattr(u, 'ocr_pages')
        )
        
        category_counts = {}
        for task in tasks:
            cat = task.get('category', 'general')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        stats = {
            'plan_id': latest_plan.id,
            'total_files': total_files,
            'total_chunks': total_chunks,
            'total_pages': total_pages,
            'total_hours': round(total_hours, 1),
            'generated_at': latest_plan.created_at,
            'categories': category_counts,
            'sort_method': metadata.get('sort_method', 'hybrid'),
            'utilization': metadata.get('utilization_percent', 0),
            'ocr_pages_processed': ocr_pages_total,
            'llm_ranked': metadata.get('llm_ranked', False),
            'ai_extraction_used': metadata.get('ai_extraction_used', False)
        }
        
        kb_stats = {
            'tasks_with_kb': 0,
            'avg_kb_relevance': 0,
            'avg_kb_confidence': 0,
            'high_confidence_tasks': 0,
            'knowledge_gaps': 0
        }
        
        kb_relevances = []
        kb_confidences = []
        
        for task in tasks:
            kb_rel = task.get('kb_relevance', 0)
            kb_conf = task.get('kb_confidence', 0)
            kb_depth = task.get('kb_depth', 'unknown')
            
            if kb_conf > 0:
                kb_stats['tasks_with_kb'] += 1
                kb_relevances.append(kb_rel)
                kb_confidences.append(kb_conf)
                
                if kb_conf > 0.7:
                    kb_stats['high_confidence_tasks'] += 1
                
                if kb_depth in ['minimal', 'none', 'limited']:
                    kb_stats['knowledge_gaps'] += 1
        
        if kb_relevances:
            kb_stats['avg_kb_relevance'] = round(sum(kb_relevances) / len(kb_relevances), 3)
            kb_stats['avg_kb_confidence'] = round(sum(kb_confidences) / len(kb_confidences), 3)
        
        stats['kb_grounding'] = kb_stats
        
    else:
        tasks = []
        kanban_columns = {'high_quality': [], 'back_log': [], 'validated': []}
        schedule = None
        stats = None
    
    return render(request, 'core/result_agentic.html', {
        'tasks': tasks,
        'kanban_columns': kanban_columns,
        'schedule': schedule,
        'stats': stats,
        'request': request
    })


@login_required
def history_detail(request, history_id):
    """View specific plan"""
    history = StudyPlanHistory.objects.filter(
        id=history_id,
        user=request.user
    ).prefetch_related('uploads').first()
    
    if not history:
        messages.error(request, 'History not found')
        return redirect('upload_page_optimized')
    
    plan_data = history.plan_json
    schedule = history.get_schedule()
    tasks = history.get_tasks()
    
    # --- NEW KANBAN LOGIC ---
    kanban_columns = _bucket_tasks_kanban(tasks)
    # ------------------------
    
    stats = {
        'id': history.id,
        'total_files': history.total_files,
        'total_pages': history.total_pages,
        'total_chunks': history.total_chunks,
        'total_hours': round(history.total_hours, 1) if history.total_hours > 0 else 0,
        'generated_at': history.created_at,
        'sort_method': history.sort_method,
        'execution_time': history.execution_time,
        'tool_calls': history.tool_calls,
        'ocr_pages_processed': history.ocr_pages_total,
        'user_goal': history.user_goal,
        'constraints': history.constraints,
    }
    
    if schedule:
        total_scheduled = sum(s.get('hours', 0) for s in schedule)
        stats['utilization'] = (total_scheduled / history.total_hours * 100) if history.total_hours > 0 else 0
    else:
        stats['utilization'] = 0
    
    return render(request, 'core/history_detail.html', {
        'history': history,
        'tasks': tasks,
        'kanban_columns': kanban_columns,
        'schedule': schedule,
        'stats': stats,
        'request': request
    })


@login_required
@require_http_methods(["POST"])
def delete_history(request, history_id):
    """Soft-delete a StudyPlanHistory by marking its status as 'deleted'.

    Only the owner may delete their history. Uses POST to avoid accidental GET deletions.
    """
    history = StudyPlanHistory.objects.filter(id=history_id, user=request.user).first()
    if not history:
        messages.error(request, 'History not found or access denied.')
        return redirect('upload_page_optimized')

    # Soft delete by updating status; keep record for analytics/audit
    history.status = 'deleted'
    history.save(update_fields=['status'])
    messages.success(request, 'Study plan history deleted.')
    return redirect('upload_page_optimized')


@login_required
def agent_logs(request):
    """Admin logs"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied.')
        return redirect('upload_page_optimized')
    
    logs = ToolLogger.get_logs()
    summary = ToolLogger.get_summary()
    
    recent_plans = Plan.objects.select_related('user').order_by('-created_at')[:20]
    
    plan_stats = []
    for plan in recent_plans:
        plan_data = {
            'id': plan.id,
            'user': plan.user.username,
            'created': plan.created_at,
            'tasks': len(plan.plan_json),
            'has_schedule': any(
                item.get('file') in ['WEEKLY SCHEDULE', 'WEEKLY SCHEDULE']
                for item in plan.plan_json
            )
        }
        plan_stats.append(plan_data)
    
    return render(request, 'core/agent_logs.html', {
        'logs': logs,
        'summary': summary,
        'recent_plans': plan_stats,
        'performance_metrics': ToolLogger.performance_metrics
    })


def register_page(request):
    form = RegistrationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        messages.success(request, 'Account created. Please log in.')
        return redirect('login_page')
    return render(request, 'core/register.html', {'form': form})

def login_page(request):
    form = LoginForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            messages.success(request, f'Welcome, {username}')
            return redirect('upload_page_optimized')
        messages.error(request, 'Invalid credentials.')
    return render(request, 'core/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'Logged out successfully.')
    return redirect('login_page')