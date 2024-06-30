import os
from dotenv import load_dotenv
import openai

class keysInstance:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAIKEY")
        self.pinecone_api_key = os.getenv("PCKEY")
        self.pinecone_data_url = os.getenv("PCDATAURL")
        self.index_name = os.getenv("INDEXNAME")
        self.email = os.getenv("EMAIL")
        self.password = os.getenv("PASSWORD")

        if not self.openai_api_key:
            raise ValueError("OPENAIKEY is not set in the environment variables")
        if not self.pinecone_api_key:
            raise ValueError("PCKEY is not set in the environment variables")
        if not self.pinecone_data_url:
            raise ValueError("PCDATAURL is not set in the environment variables")
        if not self.index_name:
            raise ValueError("INDEXNAME is not set in the environment variables")
        if not self.email:
            raise ValueError("EMAIL is not set in the environment variables")
        if not self.password:
            raise ValueError("PASSWORD is not set in the environment variables")

        openai.api_key = self.openai_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

    def get_openai_api_key(self):
        return self.openai_api_key

    def get_pinecone_api_key(self):
        return self.pinecone_api_key
    
    def get_pinecone_data_url(self):
        return self.pinecone_data_url
    
    def get_index_name(self):
        return self.index_name

    def get_password_api_key(self):
        return self.password

    def get_email_api_key(self):
        return self.email

