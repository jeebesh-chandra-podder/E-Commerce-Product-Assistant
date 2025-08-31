import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config

class DataIngestion:
    """
    Class to handle data transformation and ingestion into AstraDB vector store.
    """
    def __init__(self):
        pass
    
    def _load_env_variables(self):
        pass
        
    def _get_csv_path(self):
        pass
    
    def _load_csv(self):
        pass
    
    def transform_data(self):
        pass
    
    def store_in_vector_db(self):
        pass
    
    def run_pipeline(self):
        pass
    