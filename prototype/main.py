import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings

TRAIN_FILENAME = './train.parquet'

def main():
    clear = "\033[2J\033[H"
    lowerleft = "\033[9999;1H"
    table = pq.read_table(TRAIN_FILENAME).to_pandas()
    for row in range(table.shape[0]):
        question = table[['question']].iat[row, 0]
        answer = table[['provided_answer']].iat[row, 0]
        reference = table[['reference_answer']].iat[row, 0]
        score = table[['normalized_grade']].iat[row, 0]
        print(f'{clear}Q: {question}\nA: {answer}\nR: {reference}\nS: {score}\n')
        print(f'{lowerleft}{row + 1} / {table.shape[0]}', end='', flush=True)

if __name__ == "__main__":
    main()
