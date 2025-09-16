import os
import sys
import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import ProductAssistantException
import asyncio

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
This class is like a security guard for your secret keys. Its only job is to find, verify, and provide API keys when needed. 
This is a great practice called Separation of Concerns—you have one piece of code dedicated to one specific job.
'''
class ApiKeyManager:
    
    '''
    This defines a list of API keys that are absolutely necessary for the application to run. 
    If any of these are missing, the app should stop immediately. 
    This is much better than letting it run and then crash later when it tries to use a key that isn't there.
    '''
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY"]

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    This is the constructor (__init__) method, which runs automatically when you create an ApiKeyManager
    '''
    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        '''
        It first looks for a single environment variable called API_KEYS. 
        The idea is that in a production environment (like a server or a cloud service like AWS ECS), 
        you might pass all your secrets together as a single JSON string.
        '''
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        '''
        This is a fallback mechanism, which is excellent design. 
        If a key wasn't found in the API_KEYS JSON blob, it then tries to find it as a standalone environment variable (e.g., GOOGLE_API_KEY=...). 
        This makes the code flexible—it works one way in production (JSON blob) and another way for local development (from a .env file).
        '''
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        '''
        This is the final, crucial check. After trying both methods, it confirms that all keys from the REQUIRED_KEYS list have been found.
        -> If any are missing, it logs a serious error and raises an exception. This stops the program with a clear message, which is exactly what you want.
        -> The last line logs that the keys were loaded, but it cleverly only shows the first 6 characters (v[:6] + "..."). 
        This is a critical security practice. You never want to print entire secret keys to your logs, where they could be exposed.
        '''
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise ProductAssistantException("Missing API keys", sys)

        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    This is a simple "getter" method. Other parts of the code will call api_key_mgr.get("GOOGLE_API_KEY") to safely retrieve a key.
    If the key doesn't exist for some reason, it raises an error.
    '''
    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class ModelLoader:
    '''
    This is the main class of the file. It uses the ApiKeyManager and your configuration files to prepare the AI models.
    '''

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    It checks an environment variable ENV. 
    If the environment is not production, it calls load_dotenv(). 
    This is the magic that loads your .env file for local development. 
    In production, you'd set the ENV variable to production, and this step would be skipped.

    It then creates an instance of our ApiKeyManager to handle the keys.

    Finally, it calls load_config() to load the rest of the application settings (like which model names to use, temperature, etc.) from a file.
    '''
    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    This method loads the embedding model. An embedding model's job is to turn text (like "hello world") into a list of numbers (a vector). 
    This numerical representation helps the AI understand the meaning and context of the text.

    It gets the model name from the config file you loaded earlier.

    The "asyncio" part is a technical but important patch. The underlying Google library sometimes needs something called an "event loop" to be running. 
    This code safely checks if one exists and creates one if it doesn't, preventing a common error. This shows attention to detail.
    '''
    def load_embeddings(self):
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            # Patch: Ensure an event loop exists for gRPC aio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")  # type: ignore
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ProductAssistantException("Failed to load embedding model", sys)


    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    This method loads the main Language Model (LLM)—the one that does the chatting and text generation.

    It's designed to be flexible. It reads an environment variable LLM_PROVIDER to decide which model to load ('google', 'groq', etc.). If that variable isn't set, it defaults to 'google'.

    It then reads the specific settings for that provider from your config file (model name, temperature, etc.).

    Using an if/elif/else block, it instantiates the correct object (ChatGoogleGenerativeAI or ChatGroq).
    '''
    def load_llm(self):

        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "google")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")