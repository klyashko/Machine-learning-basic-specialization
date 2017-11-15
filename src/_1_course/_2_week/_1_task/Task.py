import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.linalg import solve
from src.utils.Utils import save_answer

sent = 'Sentence'
words = 'Words'

with open('data/sentences.txt') as f:
    data = pd.DataFrame({sent: f.readlines()})

data[sent] = data[sent].apply(lambda x: x.lower())
data[words] = data[sent].apply(lambda x: [w for w in re.split('[^a-z]', x) if w != ''])

words_dict = {}
for words_array in data[words].values:
    for w in words_array:
        if w not in words_dict:
            words_dict[w] = len(words_dict)

words_matrix = []
for words_array in data[words].values:
    temp_line = np.zeros(len(words_dict))
    for w in words_array:
        temp_line[words_dict[w]] = temp_line[words_dict[w]] + 1
    words_matrix.append(temp_line)

words_matrix = np.array(words_matrix)

result = []

for i in range(1, words_matrix.shape[0]):
    result.append(cosine(words_matrix[0], words_matrix[i]))

result = pd.Series(result, index=range(1, words_matrix.shape[0]))

save_answer(1, result.sort_values()[:2].index.values, space=True)


def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


def polynomial(array):
    vector = []
    matrix = []
    for x in array:
        vector.append(f(x))
        row = []
        for i in range(0, len(array)):
            row.append(x ** i)
        matrix.append(row)

    return np.array(vector), np.array(matrix)


vector, matrix = polynomial([1., 15.])
print(solve(matrix, vector))
vector, matrix = polynomial([1., 8., 15.])
print(solve(matrix, vector))
vector, matrix = polynomial([1., 4., 10., 15.])
answers = [round(x, 2) for x in solve(matrix, vector)]
save_answer(2, answers, space=True)
