# core/views_optimized.py - Optimized Views with History Tracking

from django.contrib.auth import authenticate, login, logout
from .forms import LoginForm, RegistrationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import Upload, Plan, Chunk, StudyPlanHistory
from .tasks import process_upload
from .tasks_agentic_optimized import generate_optimized_plan_async
from .agent_tools_advanced import ToolLogger
from celery.result import AsyncResult
from typing import List, Dict
import json
import time

User = get_user_model()

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_WAIT_TIME = 300  # 5 minutes for PDF processing

# ==================== HELPER FUNCTION ====================
def safe_int(value, default=0):
    """Safely convert to int, handling empty strings"""
    if value is None or value == '':
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# ==================== OPTIMIZED UPLOAD ====================
@login_required
def upload_page_optimized(request):
    """Optimized upload with chunked processing and history tracking"""
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        
        # Extract form data
        user_goal = request.POST.get('user_goal', 'finish semester with good grades').strip()
        sort_method = request.POST.get('sort_method', 'hybrid')
        constraint_prompt = request.POST.get('constraint_prompt', '').strip()
        
        # Flexible time input - handle empty strings
        time_input = {
            'years': safe_int(request.POST.get('years')),
            'months': safe_int(request.POST.get('months')),
            'weeks': safe_int(request.POST.get('weeks')),
            'days': safe_int(request.POST.get('days')),
            'hours': safe_int(request.POST.get('hours'))
        }
        
        # Validate
        if not files:
            messages.error(request, 'Please select at least one PDF file.')
            return render(request, 'core/upload.html', {})
        
        if len(files) > 10:
            messages.error(request, 'Maximum 10 files allowed.')
            return render(request, 'core/upload.html', {})
        
        # Validate file sizes
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
        
        # Process uploads
        upload_ids = []
        new_uploads = []
        total_size = 0
        
        for file in files:
            file_size_mb = file.size / (1024 * 1024)
            total_size += file_size_mb
            
            # Check for existing processed upload (cache reuse)
            existing = Upload.objects.filter(
                user=request.user,
                filename=file.name,
                status='processed',
                pages__isnull=False
            ).first()
            
            if existing:
                print(f"♻️ Cache hit: {file.name}")
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
                print(f"📤 New upload: {file.name} ({file_size_mb:.1f}MB)")
        
        # Launch async PDF processing
        for upload in new_uploads:
            process_upload.delay(upload.id)
        
        if total_size > 50:
            messages.info(
                request,
                f'Processing {len(new_uploads)} large files ({total_size:.1f}MB). '
                f'This will run in the background.'
            )
        
        # Wait briefly for fast PDFs
        if new_uploads:
            start = time.time()
            quick_timeout = min(30, MAX_WAIT_TIME)  # Max 30s for quick files
            
            while time.time() - start < quick_timeout:
                all_done = all(
                    Upload.objects.get(id=u.id).status in ['processed', 'failed']
                    for u in new_uploads
                )
                
                if all_done:
                    print(f"✅ Fast processing completed in {time.time() - start:.1f}s")
                    break
                
                time.sleep(1)
            
            # Check status
            for u in new_uploads:
                u.refresh_from_db()
                if u.status == 'failed':
                    messages.error(request, f"Failed to process: {u.filename}")
                elif u.status != 'processed':
                    messages.warning(
                        request,
                        f"Still processing: {u.filename}. Plan will be generated with available files."
                    )
        
        # Launch async agentic planning
        print(f"🚀 Launching optimized agentic planning...")
        
        task = generate_optimized_plan_async.delay(
            user_id=request.user.id,
            upload_ids=upload_ids,
            user_goal=user_goal,
            time_input=time_input,
            constraints=constraint_prompt,
            sort_method=sort_method
        )
        
        print(f"✅ Task launched: {task.id}")
        
        # Store in session
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
            f'🤖 AI analysis started for {len(files)} file(s) using "{sort_method}" sorting method.'
        )
        
        return redirect('planning_progress')
    
    # GET - show upload form with history
    user_uploads = Upload.objects.filter(
        user=request.user,
        status='processed'
    ).order_by('-created_at')[:10]
    
    # Get user's study plan history
    history_plans = StudyPlanHistory.objects.filter(
        user=request.user,
        status='active'
    ).order_by('-created_at')[:10]
    
    return render(request, 'core/upload.html', {
        'recent_uploads': user_uploads,
        'history_plans': history_plans
    })


# ==================== PLANNING PROGRESS ====================
@login_required
def planning_progress(request):
    """Show progress with real-time updates"""
    task_id = request.session.get('planning_task_id')
    params = request.session.get('planning_params', {})
    
    if not task_id:
        messages.error(request, 'No planning task found')
        return redirect('upload_page_optimized')
    
    return render(request, 'core/planning_progress.html', {
        'task_id': task_id,
        'params': params
    })


# ==================== STATUS API ====================
@login_required
@require_http_methods(["GET"])
def planning_status_api(request, task_id):
    """Fast status check API"""
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
        response["message"] = "Task queued, waiting to start..."
    elif result.state == 'FAILURE':
        response["error"] = str(result.info)
    
    return JsonResponse(response)


# ==================== RESULT PAGE ====================
@login_required
def result_page_optimized(request):
    """Display results with enhanced visualization"""
    latest_plan = Plan.objects.filter(
        user=request.user
    ).order_by('-created_at').first()
    
    if latest_plan:
        plan = latest_plan.plan_json
        
        # Get associated uploads for OCR info
        uploads = Upload.objects.filter(
            user=request.user,
            status='processed'
        ).order_by('-created_at')[:10]
        
        # Separate schedule from tasks
        schedule = None
        tasks = []
        metadata = {}
        
        for item in plan:
            if item.get('file') == '📅 WEEKLY SCHEDULE':
                schedule = item.get('schedule', [])
                metadata = item.get('metadata', {})
            else:
                tasks.append(item)
        
        # Calculate statistics
        total_files = len(tasks)
        total_chunks = sum(p.get('chunk_count', 0) for p in tasks)
        total_pages = sum(p.get('pages', 0) for p in tasks)
        total_hours = sum(p.get('estimated_hours', 0) for p in tasks)
        
        # OCR statistics
        ocr_pages_total = sum(
            u.ocr_pages for u in uploads if hasattr(u, 'ocr_pages')
        )
        
        # Category breakdown
        category_counts = {}
        for task in tasks:
            cat = task.get('category', 'general')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        stats = {
            'total_files': total_files,
            'total_chunks': total_chunks,
            'total_pages': total_pages,
            'total_hours': round(total_hours, 1),
            'generated_at': latest_plan.created_at,
            'categories': category_counts,
            'sort_method': metadata.get('sort_method', 'hybrid'),
            'utilization': metadata.get('utilization_percent', 0),
            'ocr_pages_processed': ocr_pages_total
        }
    else:
        tasks = []
        schedule = None
        stats = None
    
    return render(request, 'core/result_agentic.html', {
        'tasks': tasks,
        'schedule': schedule,
        'stats': stats
    })


# ==================== HISTORY DETAIL VIEW ====================
@login_required
def history_detail(request, history_id):
    """View a specific plan from history"""
    history = StudyPlanHistory.objects.filter(
        id=history_id,
        user=request.user
    ).prefetch_related('uploads').first()
    
    if not history:
        messages.error(request, 'History not found')
        return redirect('upload_page_optimized')
    
    # Extract data
    plan_data = history.plan_json
    schedule = history.get_schedule()
    tasks = history.get_tasks()
    
    # Build stats
    stats = {
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
    
    # Calculate utilization
    if schedule:
        total_scheduled = sum(s.get('hours', 0) for s in schedule)
        stats['utilization'] = (total_scheduled / history.total_hours * 100) if history.total_hours > 0 else 0
    else:
        stats['utilization'] = 0
    
    return render(request, 'core/history_detail.html', {
        'history': history,
        'tasks': tasks,
        'schedule': schedule,
        'stats': stats
    })


# ==================== ADMIN: AGENT LOGS ====================
@login_required
def agent_logs(request):
    """View detailed agent logs (admin only)"""
    if not request.user.is_staff:
        messages.error(request, 'Access denied. Admin only.')
        return redirect('upload_page_optimized')
    
    logs = ToolLogger.get_logs()
    summary = ToolLogger.get_summary()
    
    # Get recent plans with metadata
    recent_plans = Plan.objects.select_related('user').order_by('-created_at')[:20]
    
    # Performance analysis
    plan_stats = []
    for plan in recent_plans:
        plan_data = {
            'id': plan.id,
            'user': plan.user.username,
            'created': plan.created_at,
            'tasks': len(plan.plan_json),
            'has_schedule': any(
                item.get('file') == '📅 WEEKLY SCHEDULE'
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


# ==================== AUTH VIEWS ====================
def register_page(request):
    form = RegistrationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        messages.success(request, f'Account created! Please log in.')
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
            messages.success(request, f'Welcome back, {username}!')
            return redirect('upload_page_optimized')
        messages.error(request, 'Invalid credentials.')
    return render(request, 'core/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'Logged out successfully.')
    return redirect('login_page')