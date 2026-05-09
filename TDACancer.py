import networkx as nx
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
import math
import json

# ======================================================
# GRAPH FILES
# ======================================================

GRAPH_FILES = {
    "cancer": "CancerGraph/cancer_network.gml",
    "healthy": "HealthyGraph/healthy_network.gml"
}

# ======================================================
# CLEAN GRAPH
# ======================================================


def clean_graph(G):

    for u, v, data in G.edges(data=True):

        w = data.get("weight", 1.0)

        try:
            w = float(w)
        except:
            w = 1.0

        if math.isnan(w) or math.isinf(w) or w < 0:
            w = 1.0

        w = max(0.0, min(w, 1.0))

        data["weight"] = w

    return G

# ======================================================
# DISTANCE MATRIX
# ======================================================


def weighted_distance_matrix(G, weight_attr="weight", use_weights=True):

    H = G.copy()

    if use_weights:

        for u, v, d in H.edges(data=True):

            raw = d.get(weight_attr, 0.7)
            d["length"] = 1.0 - (raw / 1.0)

        lengths = dict(
            nx.all_pairs_dijkstra_path_length(
                H,
                weight="length"
            )
        )

    else:

        lengths = dict(nx.all_pairs_shortest_path_length(H))

    nodes = list(H.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    D = np.full((N, N), np.inf)

    for u, dists in lengths.items():
        for v, d in dists.items():
            D[idx[u]][idx[v]] = d

    finite_max = D[np.isfinite(D)].max()
    D[~np.isfinite(D)] = 2 * finite_max

    np.fill_diagonal(D, 0)

    return D, nodes

# ======================================================
# PERSISTENCE ENTROPY
# ======================================================


def persistence_entropy(dgm, remove_inf=True):

    if remove_inf:
        dgm = dgm[np.isfinite(dgm[:, 1])]

    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    p = lifetimes / lifetimes.sum()

    return -np.sum(p * np.log(p + 1e-12))

# ======================================================
# BETTI CURVE
# ======================================================


def betti_curve(dgm, t_values, remove_inf=True):

    if remove_inf:
        dgm = dgm[np.isfinite(dgm[:, 1])]

    counts = np.array([
        np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t))
        for t in t_values
    ])

    return counts

# ======================================================
# SIGNIFICANT FEATURES
# ======================================================


def significant_features(dgm, percentile=75, remove_inf=True):

    dgm_fin = dgm[np.isfinite(dgm[:, 1])] if remove_inf else dgm

    lifetimes = dgm_fin[:, 1] - dgm_fin[:, 0]

    threshold = np.percentile(lifetimes, percentile)

    mask = lifetimes >= threshold
    return dgm_fin[mask], lifetimes[mask], threshold

# ======================================================
# SAFE JSON LIST
# ======================================================


def safe_list(arr):
    return [float(x) for x in np.array(arr).flatten()]

# ======================================================
# RUN TDA
# ======================================================


def run_tda(graph_file, graph_name):

    print("\n" + "=" * 60)
    print(f"TDA ANALYSIS: {graph_name.upper()}")
    print("=" * 60)

    G = nx.read_gml(graph_file)
    G = clean_graph(G)

    D_weighted, nodes = weighted_distance_matrix(
        G,
        use_weights=True
    )

    print(
        f"Distance matrix: {D_weighted.shape} "
        f"| max dist: {D_weighted.max():.4f}"
    )

    result = ripser(
        D_weighted,
        metric="precomputed",
        maxdim=1,
        distance_matrix=True
    )

    dgms = result["dgms"]

    H0, H1 = dgms[0], dgms[1]

    pe_H0 = persistence_entropy(H0)
    pe_H1 = persistence_entropy(H1)

    t = np.linspace(0, D_weighted.max() * 0.1, 300)

    b0 = betti_curve(H0, t)
    b1 = betti_curve(H1, t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t, b0, lw=2, label="β0")
    axes[0].plot(t, b1, lw=2, label="β1")

    axes[0].set_title(f"Betti Curves ({graph_name})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    plot_diagrams(dgms, ax=axes[1], show=False)

    axes[1].set_title(
        f"Persistence Diagram ({graph_name})"
    )

    plt.tight_layout()

    plt.savefig(
        f"tda_{graph_name}.png",
        dpi=150
    )

    plt.close()

    sig_H1, lives_H1, thresh_H1 = significant_features(H1)

    tda_results = {

        "betti_numbers": {
            "H0_count": int(len(H0)),
            "H1_count": int(len(H1))
        },

        "persistence_entropy": {
            "H0": float(pe_H0),
            "H1": float(pe_H1)
        },

        "betti_curves": {
            "thresholds": safe_list(t),
            "beta_0": safe_list(b0),
            "beta_1": safe_list(b1)
        },

        "significant_H1_cycles": [
            {
                "birth": float(b),
                "death": float(d),
                "lifetime": float(d - b)
            }
            for (b, d) in sig_H1
        ],

        "topological_summary": {
            "nodes": int(len(nodes)),
            "H0_components": int(len(H0)),
            "H1_cycles": int(len(H1)),
            "significant_cycles": int(len(sig_H1))
        }
    }

    output_file = f"{graph_name}_tda_results.json"

    with open(output_file, "w") as f:
        json.dump(tda_results, f, indent=4)

    print(f"Saved: {output_file}")

    return tda_results


# ======================================================
# RUN BOTH DATASETS
# ======================================================

all_results = {}

for dataset_name, graph_file in GRAPH_FILES.items():

    all_results[dataset_name] = run_tda(
        graph_file,
        dataset_name
    )

# ======================================================
# COMPARISON
# ======================================================

comparison = {

    "cancer": all_results["cancer"],
    "healthy": all_results["healthy"],

    "comparative_topology": {

        "H0_difference": (
            all_results["cancer"]["betti_numbers"]["H0_count"]
            -
            all_results["healthy"]["betti_numbers"]["H0_count"]
        ),

        "H1_difference": (
            all_results["cancer"]["betti_numbers"]["H1_count"]
            -
            all_results["healthy"]["betti_numbers"]["H1_count"]
        ),

        "entropy_H0_difference": (
            all_results["cancer"]["persistence_entropy"]["H0"]
            -
            all_results["healthy"]["persistence_entropy"]["H0"]
        ),

        "entropy_H1_difference": (
            all_results["cancer"]["persistence_entropy"]["H1"]
            -
            all_results["healthy"]["persistence_entropy"]["H1"]
        )
    }
}

with open("comparative_tda_analysis.json", "w") as f:
    json.dump(comparison, f, indent=4)

print("\nSaved comparative_tda_analysis.json")