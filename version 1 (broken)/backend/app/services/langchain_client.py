from langchain_community.chat_models import ChatOpenAI

def get_llm():
    return ChatOpenAI(model_name="gpt-5-mini", temperature=0)
