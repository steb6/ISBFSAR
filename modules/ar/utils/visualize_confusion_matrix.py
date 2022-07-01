import pickle as pkl
import numpy as np

with open("CONFUSIONMATRIX", "rb") as infile:
    data = pkl.load(infile)

matrix = np.zeros((17, 17))
cs = list(data.keys())

best = np.array([3, 2, 5, 7, 10, 11, 13, 15])

cs = [cs[elem] for elem in best]
matrix = matrix[best][:, best]

for i, ss in enumerate(range(matrix.shape[0])):
    for j, q in enumerate(range(matrix.shape[1])):
        matrix[i][j] = data[cs[i]][cs[j]]
data