from typing import List
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render, redirect
from django.core.files.uploadedfile import UploadedFile
from .models import UploadedDocument, ProcessingRun
from .utils.pipeline import process_documents
import os


def home(request: HttpRequest) -> HttpResponse:
    latest_run = ProcessingRun.objects.order_by('-created_at').first()
    context = {"latest_run": latest_run}
    return render(request, 'assistant/home.html', context)


def upload_files(request: HttpRequest) -> HttpResponse:
    if request.method == 'POST':
        files: List[UploadedFile] = request.FILES.getlist('files')
        created_docs: List[UploadedDocument] = []
        for f in files[:5]:
            doc = UploadedDocument.objects.create(
                original_name=f.name,
                file=f,
            )
            created_docs.append(doc)
        request.session['last_doc_ids'] = [d.id for d in created_docs]
        return redirect('assistant:home')
    return render(request, 'assistant/upload.html')


def run_pipeline(request: HttpRequest) -> HttpResponse:
    prompt = request.POST.get('prompt', '') if request.method == 'POST' else ''
    doc_ids = request.session.get('last_doc_ids', [])
    documents = list(UploadedDocument.objects.filter(id__in=doc_ids))
    ranking, summaries, quiz = process_documents(documents, prompt)
    run = ProcessingRun.objects.create(
        user_prompt=prompt,
        ranking_json=ranking,
        summary_json=summaries,
        quiz_json=quiz,
    )
    run.documents.set(documents)
    return redirect('assistant:home')

