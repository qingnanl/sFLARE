# src/yourpkg/mnn.py
from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm

def find_mutual_nn_fast(data1, data2, k1=30, k2=30, n_jobs=8, progress=True):
    n1, n2 = data1.shape[0], data2.shape[0]
    tree1 = cKDTree(data1)
    tree2 = cKDTree(data2)
    k_index_1 = tree1.query(x=data2, k=k1, workers=n_jobs)[1]  # shape: (n2, k1) -> indices into data1
    k_index_2 = tree2.query(x=data1, k=k2, workers=n_jobs)[1]  # shape: (n1, k2) -> indices into data2
    inv = [[] for _ in range(n2)]
    for i1 in range(n1):
        for i2 in k_index_2[i1]:
            inv[i2].append(i1)
    for i2 in range(n2):
        if inv[i2]:
            inv[i2] = np.array(sorted(inv[i2]), dtype=np.int32)
        else:
            inv[i2] = np.empty(0, dtype=np.int32)
    mutual_pairs = []
    rng = range(n2)
    if progress:
        rng = tqdm(rng, desc="Mutual NN (scan)", unit="cell")
    for i2 in rng:
        # candidates: i1 among k1-NN of i2
        a = k_index_1[i2]
        # i1 that have i2 among their k2-NN
        b = inv[i2]
        if b.size == 0:
            continue
        common = np.intersect1d(a, b, assume_unique=False)
        if common.size:
            mutual_pairs.extend(zip(common.tolist(), [i2]*common.size))
    return mutual_pairs