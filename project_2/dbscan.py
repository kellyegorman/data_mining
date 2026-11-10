import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from Preprocessing import l_data,p_data
for n in["cho", "iyer"]:
    x,y=l_data(f"data/{n}.txt")
    preprocessor = p_data(x, scale=True)
    process=preprocessor.fit_transform(x)
    if n == "cho":
        eps_val=[2.8,3.0,3.2,3.5]
    else:
        eps_val=[2.0,2.5,3.0,3.5]
    min_samples=[3,5,10,15]
    for i in eps_val:
        for j in min_samples:
            model= DBSCAN(eps=i,min_samples=j)
            predicted_val= model.fit_predict(process)
            cluster=len(set(predicted_val) -{-1})
            noise = sum(predicted_val == -1)
            ars= adjusted_rand_score(y, predicted_val)
            print(f"\n Dataset {n}.txt with eps={i}, min_samples={j}")
            print ("cluster Found", cluster)
            print("Noise:", noise)
            print("Adjusted Rand Score:", ars)

