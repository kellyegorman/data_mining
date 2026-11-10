import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import pair_confusion_matrix
try:
    from Preprocessing import l_data, p_data
except ImportError:
    from sklearn.preprocessing import StandardScaler

    def l_data(path):
        df = pd.read_csv(path, sep=r"\s+", header=None)
        return df.iloc[:, 2:].to_numpy(dtype=float), df.iloc[:, 1].to_numpy(dtype=int)
    def p_data(x, scale=True):
        return StandardScaler()

DATASETS = ["cho", "iyer"]
LINKAGES = ["ward", "average"]  
RANDOM_STATE = 42

def external_indices(truth, pred):
    pc = pair_confusion_matrix(truth, pred)
    tn, fp, fn, tp = pc[0, 0], pc[0, 1], pc[1, 0], pc[1, 1]
    return {
        "Rand": (tp + tn) / (tp + tn + fp + fn),
        "ARI": adjusted_rand_score(truth, pred),
        "Jaccard": tp / (tp + fp + fn),
    }

def plot_pca(coords, panels, title, explained):
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab20")
    for ax, (ptitle, labels) in zip(axes, panels):
        for i, lab in enumerate(sorted(set(labels))):
            m = labels == lab
            if lab == -1:
                ax.scatter(coords[m, 0], coords[m, 1], s=12, c="lightgray",
                           label="outlier (-1)")
            else:
                ax.scatter(coords[m, 0], coords[m, 1], s=12, color=cmap(i % 20),
                           label=f"cluster {lab}")
        ax.set_title(ptitle)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
        ax.legend(fontsize=6, markerscale=1.5, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()


def main():
    for n in DATASETS:
        x, y = l_data(f"data/{n}.txt")
        data = p_data(x, scale=True).fit_transform(x)
        k = len(set(y) - {-1})
        print(f"\n=== HAC on {n}.txt: {data.shape[0]} genes, "
              f"{data.shape[1]} features, {k} true clusters, "
              f"{np.sum(y == -1)} outliers ===")

        preds = {}
        summary = []
        for link in LINKAGES:
            pred = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(data)
            preds[link] = pred
            s = external_indices(y, pred)
            summary.append({
                "Linkage": link,
                "Clusters": len(set(pred)),
                "Rand": s["Rand"],
                "ARI": s["ARI"],
                "Jaccard": s["Jaccard"],
                "Silhouette": silhouette_score(data, pred),
                "Largest cluster": np.bincount(pred).max(),
                "Smallest cluster": np.bincount(pred).min(),
            })
        print(pd.DataFrame(summary).round(4).to_string(index=False))
        print("\nARI for different k (true k marked with *):")
        rows = []
        for kk in range(2, 2 * k + 1):
            row = {"k": f"{kk}{'*' if kk == k else ''}"}
            for link in LINKAGES:
                p = AgglomerativeClustering(n_clusters=kk, linkage=link).fit_predict(data)
                row[f"ARI ({link})"] = adjusted_rand_score(y, p)
            rows.append(row)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca.fit_transform(data)
        panels = [("Ground truth", y)] + [(f"HAC ({l}, k={k})", preds[l]) for l in LINKAGES]
        plot_pca(coords, panels, f"{n}.txt: PCA projection", pca.explained_variance_ratio_)
    plt.show() 

if __name__ == "__main__":
    main()