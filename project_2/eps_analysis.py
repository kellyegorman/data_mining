import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from Preprocessing import l_data,p_data
def eps_value(x, n, min_samples=5):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(x)
    distance, indices = neighbors.kneighbors(x)
    sorted_dis= np.sort(distance[:, -1])
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_dis)
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.title(f"K-Distance Plot for {n}")
    plt.grid(True)
    plt.show()
    return sorted_dis
if __name__ == "__main__":
    for n in["cho", "iyer"]:
        x,y=l_data(f"data/{n}.txt")
        preprocessor = p_data(x, scale=True)
        process=preprocessor.fit_transform(x)
        eps_value(process,n,min_samples=5)