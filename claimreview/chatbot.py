from langchain_openai import ChatOpenAI
from claimreview.prompts import Prompt
from langchain.schema import ChatMessage

class Chatbot:
    def __init__(self, vectorstore, openai_api_key):
        self.vector_store = vectorstore
        self.openai_api_key = openai_api_key
    
    def get_lcel(self, query):
        llm = ChatOpenAI(
            streaming=True,
            openai_api_key=self.openai_api_key,
            model_name="gpt-4-0125-preview",
            temperature=0.0,
        )
        context_docs = self.vector_store.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in context_docs])
        prompt_text = Prompt(question=query, context=context).chatTemplate().format()
        
        # Create a list of ChatMessage objects
        messages = [ChatMessage(role="user", content=prompt_text)]
        
        response = llm.invoke(messages)
        return response if response else None
