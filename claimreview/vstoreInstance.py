import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

class vstoreInstance:
    def __init__(self, pinecone_api_key, index_name, pinecone_data_url):
        load_dotenv()  
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        self.pinecone_data_url = pinecone_data_url
        self.pc = Pinecone(api_key=pinecone_api_key)

    def get_index(self):
        index = self.pc.Index(host=self.pinecone_data_url)
        return index

    def get_vector_store(self,index, embedder, text_field):
        vectorstore = PineconeVectorStore(index, embedder, text_field)
        return vectorstore