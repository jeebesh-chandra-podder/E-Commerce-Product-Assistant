import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
A class is like a blueprint for creating objects. By grouping all the related functions (we call them methods inside a class) and variables together, we create a neat, reusable component. 
This DataIngestion class is a blueprint for an object whose entire job is to handle the data ingestion process.
'''
class DataIngestion:

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    The __init__ method is special. It's the constructor, and it runs automatically whenever you create a new DataIngestion object. 
    Its job is to set up everything the object needs to get started.
    '''
    def __init__(self):
        print("Initializing DataIngestion pipeline...")
        self.model_loader=ModelLoader()

        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()

        self.config=load_config()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _load_env_variables(self):
        
        '''
        This is the function from the dotenv library. It looks for a .env file in your project folder and loads the key-value pairs from it into the environment.
        '''
        load_dotenv()
        
        required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]
        
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")


    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    These two methods work together. First, _get_csv_path builds the full path to your data file. 
    Again, it includes a crucial check with os.path.exists to make sure the file is actually there before trying to open it.
    '''
    def _get_csv_path(self):
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir,'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    Just like with the environment variables, the code checks if the CSV contains all the columns it expects. 
    This prevents errors that could happen if someone provided a CSV with missing or misspelled columns. This is a sign of robust, production-ready code.
    '''
    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id','product_title', 'rating', 'total_reviews','price', 'top_reviews'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    '''
    The goal here is to convert your raw data from the CSV into the special Document format that LangChain uses.
    The code iterates through each row of your product data.

    For each product, it splits the information into two parts:
        -> page_content: This is the main text you want the AI to search through. In this case, it's the top_reviews. This is the "what."
        -> metadata: This is all the other information about the text, like the product's ID, title, rating, and price. This is the "where" and "who."
    Finally, it creates a Document object for each product and returns a list of them.
    '''
    def transform_data(self):
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                    "product_id": row["product_id"],
                    "product_title": row["product_title"],
                    "rating": row["rating"],
                    "total_reviews": row["total_reviews"],
                    "price": row["price"],
                    "top_reviews": row["top_reviews"]
                }
            product_list.append(product_entry)

        documents = []
        for entry in product_list:
            metadata = {
                    "product_id": entry["product_id"],
                    "product_title": entry["product_title"],
                    "rating": entry["rating"],
                    "total_reviews": entry["total_reviews"],
                    "price": entry["price"]
            }
            doc = Document(page_content=entry["top_reviews"], metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        return documents

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    This method takes the list of Document objects and sends them to the AstraDB vector database.
    '''
    def store_in_vector_db(self, documents: List[Document]):
        collection_name=self.config["astra_db"]["collection_name"]
        vstore = AstraDBVectorStore(
            embedding= self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    This is the main orchestrator. It calls the other methods in the correct order.

    transform_data() to prepare the documents.
    store_in_vector_db() to save them.

    Excellent Test: It then performs a similarity_search. This is the whole reason you built this pipeline! 
    It takes a text query, converts that query into a vector using the same embedding model, and then asks the database to find the documents with the most similar vectors. 
    This is how you find relevant product reviews even if they don't contain the exact words "low budget iphone."
    '''
    def run_pipeline(self):
        documents = self.transform_data()
        vstore, _ = self.store_in_vector_db(documents)

        #Optionally do a quick search
        query = "Can you tell me the low budget iphone?"
        results = vstore.similarity_search(query)

        print(f"\nSample search results for query: '{query}'")
        for res in results:
            print(f"Content: {res.page_content}\nMetadata: {res.metadata}\n")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

# Run if this file is executed directly
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()