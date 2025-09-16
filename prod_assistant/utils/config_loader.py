# utils/config_loader.py
from pathlib import Path
import os
import yaml

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
This is a helper function (programmers often use a single underscore _ at the beginning of a function name to signal that it's for internal use within this file). 
Its only job is to find the absolute path to the main project folder.
'''
def _project_root() -> Path:
    # .../utils/config_loader.py -> parents[1] == project root
    '''
    -> __file__: This is a special Python variable that holds the path to the current file (config_loader.py).
    -> Path(__file__): This turns that string path into a Path object.
    -> .resolve(): This converts the path into a full, absolute path, like C:\Users\YourName\MyProject\utils\config_loader.py.
    -> .parents[1]: This is the clever part. .parents gives you a list of all the parent directories.
        -> parents[0] would be the immediate parent: the utils folder.
        -> parents[1] is the parent of utils: the main MyProject folder. This is our project root!
    '''
    return Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''
This is the main function of the script. It's the one that other parts of your application will call to get the configuration settings.

The function is defined to take an optional argument config_path. If you don't provide this argument, the function will try to find the config file in a few different ways:
1. It first checks if there's an environment variable named CONFIG_PATH. If this variable exists, it uses that as the path to the config file.
2. If the environment variable doesn't exist, it defaults to looking for a file named config.yaml inside a config folder located at the root of your project.   
'''
def load_config(config_path: str | None = None) -> dict:
    """
    Resolve config path reliably irrespective of CWD.
    Priority: explicit arg > CONFIG_PATH env > <project_root>/config/config.yaml
    """
    env_path = os.getenv("CONFIG_PATH")
    if config_path is None:
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}