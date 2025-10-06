# core/views.py (Upload + Plan API) - Pinecone SDK ใหม่

from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import Upload, Plan
from .serializers import UploadSerializer, PlanSerializer
from .tasks import process_upload
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# LangChain LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Pinecone vector store (singleton)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vector_store = PineconeVectorStore(
    index_name="dsi314",
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

class UploadViewSet(viewsets.ModelViewSet):
    queryset = Upload.objects.all()
    serializer_class = UploadSerializer

    def perform_create(self, serializer):
        upload = serializer.save(user=self.request.user)
        process_upload.delay(upload.id)

class PlanViewSet(viewsets.ViewSet):
    @action(detail=False, methods=["post"])
    def generate(self, request):
        user = request.user
        uploads = Upload.objects.filter(user=user, status="processed")
        
        if not uploads:
            return Response({"error": "No processed uploads found"}, status=status.HTTP_400_BAD_REQUEST)

        # Use LangChain retriever for querying relevant chunks across all uploads
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "filter": {"upload_id": {"$in": [up.id for up in uploads]}}}
        )

        # Custom prompt for study plan generation
        prompt_template = """
        Generate a prioritized study plan for these PDF files based on the retrieved chunks.
        For each file, provide a priority order with reasons.
        Retrieved context: {context}
        Question: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Run the chain with a query for study plan
        query = "Generate a prioritized study plan for all uploaded PDF files, including order and reasons for priority."
        result = qa_chain.run({"query": query})

        # Parse result to JSON (assuming LLM returns JSON-like string; you may need to parse it)
        try:
            import json
            plan_json = json.loads(result)
        except:
            plan_json = {"raw_plan": result}  # Fallback if not JSON

        # Save Plan (using first upload as representative)
        Plan.objects.create(user=user, upload=uploads.first(), plan_json=plan_json)

        return Response({"plan": plan_json})