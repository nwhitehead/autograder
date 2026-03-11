import pyarrow.parquet as pq
import numpy as np
import pandas as pd

TRAIN_FILENAME = './train.parquet'

def main():
    table = pq.read_table(TRAIN_FILENAME).to_pandas()
    for row in range(table.shape[0]):
        question = table[['question']].iat[row, 0]
        answer = table[['provided_answer']].iat[row, 0]
        reference = table[['reference_answer']].iat[row, 0]
        score = table[['normalized_grade']].iat[row, 0]
        print(f'Q: {question}\nA: {answer}\nR: {reference}\nS: {score}\n')

if __name__ == "__main__":
    main()
