"""
Debug script to test the RAG chain independently.
Run: python debug_chain.py
"""
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "student_assistant.settings")
django.setup()

from core.llm_config import embeddings, llm, INDEX_NAME
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import json
import re

def test_rag_chain():
    """Test the RAG chain with simplified approach."""
    
    print("="*60)
    print("TESTING RAG CHAIN")
    print("="*60)
    
    # Initialize vector store
    print("\n1. Connecting to Pinecone...")
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    print("   ✓ Connected")
    
    # Create retriever
    print("\n2. Creating retriever...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("   ✓ Retriever created")
    
    # Test retrieval
    print("\n3. Testing document retrieval...")
    try:
        docs = retriever.invoke("")  # Use invoke instead of deprecated method
        print(f"   ✓ Retrieved {len(docs)} documents")
        
        if docs:
            print("\n   Sample document:")
            print(f"   - File: {docs[0].metadata.get('file', 'Unknown')}")
            print(f"   - Content preview: {docs[0].page_content[:100]}...")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return
    
    # Test prompt template
    print("\n4. Testing prompt template...")
    prompt_template = """Analyze these documents and create a study plan.

Documents:
{context}

Output a JSON array with file, priority, and reason for each document."""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"]
    )
    print(f"   ✓ Prompt variables: {prompt.input_variables}")
    
    # Test formatting function
    print("\n5. Testing document formatting...")
    def format_docs(docs):
        formatted = []
        for doc in docs:
            formatted.append(
                f"File: {doc.metadata.get('file', 'Unknown')}\n"
                f"Content: {doc.page_content[:200]}...\n"
            )
        return "\n---\n".join(formatted)
    
    try:
        formatted = format_docs(docs)
        print(f"   ✓ Formatted {len(docs)} documents")
        print(f"   Preview: {formatted[:150]}...")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return
    
    # Test chain building
    print("\n6. Building RAG chain...")
    try:
        def retrieve_and_format(input_query):
            docs = retriever.invoke(input_query)
            return format_docs(docs)
        
        rag_chain = (
            {"context": RunnableLambda(retrieve_and_format)}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("   ✓ Chain built successfully")
    except Exception as e:
        print(f"   ✗ Error building chain: {str(e)}")
        return
    
    # Test chain invocation - METHOD 1 (empty query for testing)
    print("\n7. Testing chain invocation (method 1: empty query)...")
    try:
        result = rag_chain.invoke("")
        print(f"   ✓ Success! Result length: {len(result)} chars")
        print(f"   Preview: {result[:200]}...")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")

    # Test chain invocation - METHOD 2 (with query)
    print("\n8. Testing chain invocation (method 2: with query)...")
    try:
        result = rag_chain.invoke("create study plan")
        print(f"   ✓ Success! Result length: {len(result)} chars")
        print(f"   Preview: {result[:200]}...")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")

    # Test chain invocation - METHOD 3 (with study-specific query)
    print("\n9. Testing chain invocation (method 3: study plan query)...")
    try:
        result = rag_chain.invoke("study plan")
        print(f"   ✓ Success! Result length: {len(result)} chars")
        print(f"   Preview: {result[:200]}...")
        
        # Bonus: Quick JSON parse test here too
        json_match = re.search(r'\[.*?\]', result, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group(0))
                print(f"   ✓ Valid JSON with {len(plan)} items")
                print(f"   Sample item: {plan[0] if plan else 'N/A'}")
            except json.JSONDecodeError:
                print(f"   ✗ Found array but not valid JSON")
        else:
            print(f"   ✗ No JSON array found in response")
            
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test manual invocation
    print("\n10. Testing manual LLM call (bypass chain)...")
    try:
        from langchain_core.messages import HumanMessage
        
        # Get docs
        docs = retriever.invoke("study plan")
        context = format_docs(docs)
        
        # Format prompt manually
        full_prompt = prompt_template.replace("{context}", context)
        
        # Call LLM
        response = llm.invoke([HumanMessage(content=full_prompt)])
        result = response.content
        
        print(f"   ✓ Success! Result length: {len(result)} chars")
        print(f"   Preview: {result[:200]}...")
        
        # Try parsing JSON
        json_match = re.search(r'\[.*?\]', result, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group(0))
                print(f"   ✓ Valid JSON with {len(plan)} items")
                print(f"   Sample item: {plan[0] if plan else 'N/A'}")
            except:
                print(f"   ✗ Found array but not valid JSON")
        else:
            print(f"   ✗ No JSON array found in response")
            
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("DEBUG TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_rag_chain()