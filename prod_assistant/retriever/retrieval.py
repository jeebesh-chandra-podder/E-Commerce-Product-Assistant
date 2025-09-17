import os
import sys
from typing import List
from dotenv import load_dotenv
from pathlib import Path

from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

from utils.config_loader import load_config
from utils.model_loader import ModelLoader

'''
Path(__file__): __file__ is a special variable that holds the path to the current script (retrieval.py). Path() turns this string into a Path object.

.resolve(): This figures out the full, absolute path to the file, removing any shortcuts like . or ...

.parents[2]: This is the cool part. It walks up the directory tree from the current file. .parents[0] would be the directory containing retrieval.py, .parents[1] would be the directory above that, and so on. This line assumes your project structure is something like project_root/src/app/retrieval.py, and it's trying to get the path to project_root.

sys.path.insert(0, str(project_root)): This takes the project root path we just found and adds it to the very beginning of the list of places Python looks for modules. The essence of this is to ensure that no matter where you run this script from, it will always be able to find your utils folder. It's a robust way to handle imports in complex projects.
'''
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Retriever:
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.model_loader=ModelLoader()
        self.config=load_config()
        self._load_env_variables()
        
        '''
        This is a common and efficient pattern called lazy loading. 
        It means we're not going to connect to the database or build the retriever object the moment we create the Retriever. 
        We'll wait until we actually need them. This can make the initial startup of your application faster.
        '''
        self.vstore = None
        self.retriever = None
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _load_env_variables(self):
        load_dotenv()
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def load_retriever(self):
        
        '''
        This is the first part of our lazy loading. The code checks, "Have we already connected to the vector store?" If self.vstore is still None, it proceeds to create the AstraDBVectorStore object, just like in the ingestion script. This ensures the connection only happens once.
        '''
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]
            self.vstore = AstraDBVectorStore(
                embedding= self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace
            )
        
        '''
        The second part of the lazy loading. If the retriever hasn't been created yet, it proceeds. It reads the top_k value from the config file, defaulting to 3 if not specified. Then, it calls as_retriever on the vector store to create the retriever object, which is optimized for fetching the top_k most relevant documents based on a query.
        '''
        if not self.retriever:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3
            retriever=self.vstore.as_retriever(search_kwargs={"k": top_k})
            print("Retriever loaded successfully.")
            return retriever
            
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    This is the simple, public-facing method that another part of your application would use. It first ensures the retriever is loaded by calling self.load_retriever(). Then, it uses the retriever's invoke method to process the query and fetch relevant documents. Finally, it returns these documents to the caller.
    '''
    def call_retriever(self,query):
        retriever=self.load_retriever()
        output=retriever.invoke(query)
        return output

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
if __name__=='__main__':
    retriever_obj = Retriever()
    user_query = "Can you suggest good budget laptops?"
    results = retriever_obj.call_retriever(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")