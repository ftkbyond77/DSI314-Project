from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import FileUpload
from .services.agent import Agent
import pdfplumber

@login_required
def index(request):
    return render(request, 'app/index.html')

@login_required
def upload_files(request):
    if request.method == "POST":
        files = request.FILES.getlist('files')[:5]  # จำกัดแค่ 5 ไฟล์สำหรับทดสอบ
        file_objs = []

        for f in files:
            content = ''
            # ตรวจสอบว่าเป็น PDF หรือไม่
            if f.name.lower().endswith('.pdf'):
                with pdfplumber.open(f) as pdf:
                    for page in pdf.pages:
                        content += page.extract_text() + '\n'
            else:
                content = f.read().decode('utf-8')

            obj = FileUpload.objects.create(
                user=request.user,
                file_name=f.name,
                file_content=content
            )
            file_objs.append(obj)

        agent = Agent(request.user.id)
        plan = agent.plan_study(file_objs)
        return render(request, 'app/plan.html', {'plan': plan})

    return render(request, 'app/upload_files.html')

@login_required
def view_plan(request):
    agent = Agent(request.user.id)
    plan = agent.memory.get_plan()
    return render(request, 'app/plan.html', {'plan': plan})

@login_required
def generate_quiz(request, file_id):
    file_obj = FileUpload.objects.get(id=file_id)
    agent = Agent(request.user.id)
    quiz = agent.generate_quiz(file_obj)
    return render(request, 'app/quiz.html', {'quiz': quiz, 'file': file_obj})
