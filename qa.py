import json
from pathlib import Path
#from pprint import pprint

from langchain.document_loaders import JSONLoader 

def load_json():
    loader = JSONLoader(
        file_path = "../fitness.json"
    )

    return loader