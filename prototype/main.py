import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

LOCAL = True
MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
TRAIN_FILENAME = './train.parquet'

class Encoder:
    def __init__(self, model_name, **kwargs):
        # Can also have trucate_dim=512 to keep sizes at 512 or something
        self.model = SentenceTransformer(MODEL_NAME, local_files_only=LOCAL)

    def encode(self, x):
        return self.model.encode(x)

    def similarity(self, a, b):
        return cos_sim(self.model.encode(a, prompt_name='query'), self.encode(b))


def main():
    model = Encoder(model_name=MODEL_NAME)
    clear = "\033[2J\033[H"
    lowerleft = "\033[9999;1H"
    table = pq.read_table(TRAIN_FILENAME).to_pandas()
    for row in range(table.shape[0]):
        question = table[['question']].iat[row, 0]
        answer = table[['provided_answer']].iat[row, 0]
        reference = table[['reference_answer']].iat[row, 0]
        score = table[['normalized_grade']].iat[row, 0]
        sim = model.similarity(answer, reference)
        print(f'{clear}Q: {question}\nA: {answer}\nR: {reference}\nS: {score}\nSIM: {sim}\n')
        print(f'{lowerleft}{row + 1} / {table.shape[0]}', end='', flush=True)

if __name__ == "__main__":
    main()
