import os
import json

def load_json(fn):
    with open(fn, 'r') as f:
        file = json.load(f)
    return file