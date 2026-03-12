import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
import msgpack

LOCAL = True
MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
TRAIN_FILENAME = './train.parquet'
CACHE_FILE = './embedding.cache'

class Encoder:
    def __init__(self, model_name, **kwargs):
        # Can also have trucate_dim=512 to keep sizes at 512 or something
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=LOCAL)
        self.cache = {}
        self.count = 0
        try:
            with open(CACHE_FILE, 'rb') as fin:
                packed = fin.read()
                self.cache = msgpack.unpackb(packed)
        except:
            pass
        print(f'Size of cache: {len(self.cache)}')

    def save_cache(self):
        packed = msgpack.packb(self.cache)
        with open(CACHE_FILE, 'wb') as fout:
            fout.write(packed)

    def encode(self, x):
        if x in self.cache:
            return np.array(self.cache[x])
        result = self.model.encode(x).tolist()
        self.cache[x] = result
        self.count += 1
        if self.count % 100 == 0:
            self.save_cache()
        return np.array(result)

    def similarity(self, a, b):
        assert isinstance(a, str)
        assert isinstance(b, str)
        return float(cos_sim(self.encode(a), self.encode(b)))

def main():
    model = Encoder(model_name=MODEL_NAME)
    use_ansi = False
    clear = "\033[2J\033[H" if use_ansi else ""
    lowerleft = "\033[9999;1H" if use_ansi else ""
    table = pq.read_table(TRAIN_FILENAME).to_pandas()
    for row in range(table.shape[0]):
        question = table[['question']].iat[row, 0]
        answer = str(table[['provided_answer']].iat[row, 0])
        # Somehow \x9d is crashing python, coming from pasting windows quotes from some editor in student answers
        answer = answer.replace('â€œ', '"').replace('â€\x9d', '"').replace('\x9d', '')
        reference = str(table[['reference_answer']].iat[row, 0])
        score = table[['normalized_grade']].iat[row, 0]
        sim = model.similarity(answer, reference)
        print(f'{clear}Q: {question}\nA: {answer}\nR: {reference}\nS: {score}\nSIM: {sim}\n')
        print(f'{lowerleft}{row + 1} / {table.shape[0]}', end='', flush=True)
    model.save_cache()

if __name__ == "__main__":
    main()
