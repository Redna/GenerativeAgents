
import hashlib
import os
import json
import pickle
from sentence_transformers import SentenceTransformer

from generative_agents.utils import generate_hash_from_signature
from generative_agents import global_state

class CachableSentenceTransformer(SentenceTransformer):
    def encode(self, *args, **kwargs):
        # merge args and kwargs to one dict
        hash_key = generate_hash_from_signature(*args, **kwargs)
        cache_dir = f".generation_cache/embed/"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_path = f"{cache_dir}/{hash_key}.pkl"

        if os.path.exists(cache_file_path):
            return pickle.load(open(cache_file_path, "rb"))

        embeddings = super().encode(*args, **kwargs)
        with open(cache_file_path, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings