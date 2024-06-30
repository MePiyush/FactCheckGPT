from langchain_openai import OpenAIEmbeddings

class Embeddings:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key

    def get_embedding_object(self):
        return OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=self.openai_api_key)