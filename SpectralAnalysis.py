import json
import math
import networkx as nx
import numpy as np



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

    edges_to_remove = []

    for u, v, data in G.edges(data=True):

        w = data.get("weight", 1.0)

        if w is None:
            data["weight"] = 1.0
            continue

        try:
            w = float(w)
        except:
            edges_to_remove.append((u, v))
            continue

        if math.isnan(w) or math.isinf(w):
            edges_to_remove.append((u, v))
            continue

        data["weight"] = w

    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

# ======================================================
# SPECTRAL ANALYSIS
# ======================================================


def spectral_analysis(graph_file, graph_name):

    print("\n" + "=" * 60)
    print(f"SPECTRAL ANALYSIS: {graph_name.upper()}")
    print("=" * 60)

    G = nx.read_gml(graph_file)

    print("Original Graph")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    G = clean_graph(G)

    largest_cc = max(nx.connected_components(G), key=len)
    Gcc = G.subgraph(largest_cc).copy()

    print("Largest Connected Component")
    print("Nodes:", Gcc.number_of_nodes())
    print("Edges:", Gcc.number_of_edges())

    A = nx.to_numpy_array(
        Gcc,
        weight="weight",
        dtype=float
    )

    A = np.nan_to_num(
        A,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    L = D - A

    print("Computing eigenvalues...")

    adj_eigs = np.linalg.eigvals(A)
    lap_eigs = np.linalg.eigvals(L)

    adj_eigs = np.real(adj_eigs)
    lap_eigs = np.real(lap_eigs)

    adj_eigs = np.sort(adj_eigs)[::-1]
    lap_eigs = np.sort(lap_eigs)

    spectral_radius = float(np.max(np.abs(adj_eigs)))

    largest_adj_eigenvalue = float(adj_eigs[0])

    spectral_gap = (
        float(adj_eigs[0] - adj_eigs[1])
        if len(adj_eigs) > 1 else 0.0
    )

    algebraic_connectivity = (
        float(lap_eigs[1])
        if len(lap_eigs) > 1 else 0.0
    )

    zero_laplacian_count = int(
        np.sum(np.isclose(lap_eigs, 0))
    )

    results = {

        "graph_info": {
            "nodes": int(Gcc.number_of_nodes()),
            "edges": int(Gcc.number_of_edges())
        },

        "spectral_properties": {

            "spectral_radius":
                spectral_radius,

            "largest_adjacency_eigenvalue":
                largest_adj_eigenvalue,

            "spectral_gap":
                spectral_gap,

            "algebraic_connectivity":
                algebraic_connectivity,

            "zero_laplacian_eigenvalues":
                zero_laplacian_count
        },

        "top_20_adjacency_eigenvalues":
            adj_eigs[:20].tolist(),

        "top_20_laplacian_eigenvalues":
            lap_eigs[:20].tolist()
    }

    output_file = f"{graph_name}_spectral_properties.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved: {output_file}")

    return results


# ======================================================
# RUN BOTH ANALYSES
# ======================================================

all_results = {}

for name, graph_file in GRAPH_FILES.items():

    all_results[name] = spectral_analysis(
        graph_file,
        name
    )

# ======================================================
# COMPARATIVE ANALYSIS
# ======================================================

comparison = {

    "cancer": all_results["cancer"],
    "healthy": all_results["healthy"],

    "comparative_metrics": {

        "spectral_radius_difference": (
            all_results["cancer"]["spectral_properties"]["spectral_radius"]
            -
            all_results["healthy"]["spectral_properties"]["spectral_radius"]
        ),

        "spectral_gap_difference": (
            all_results["cancer"]["spectral_properties"]["spectral_gap"]
            -
            all_results["healthy"]["spectral_properties"]["spectral_gap"]
        ),

        "algebraic_connectivity_difference": (
            all_results["cancer"]["spectral_properties"]["algebraic_connectivity"]
            -
            all_results["healthy"]["spectral_properties"]["algebraic_connectivity"]
        )
    }
}

with open("comparative_spectral_analysis.json", "w") as f:
    json.dump(comparison, f, indent=4)

print("\nSaved comparative_spectral_analysis.json")