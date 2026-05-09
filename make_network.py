"""
Protein Interaction Network Builder
====================================
Builds a NetworkX graph from BioGRID protein interaction data for:
  - Autism Spectrum Disorder (ASD)
  - Alzheimer's Disease (AD)

Data source: https://downloads.thebiogrid.org/BioGRID/Latest-Release/
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")          # headless backend — swap to "TkAgg" if you want a live window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

import os
import pandas as pd

# ==========================================
# FOLDER PATHS
# ==========================================
alz_folder = "data/alz"
aut_folder = "data/aut"

# ==========================================
# COLUMNS WE ACTUALLY NEED
# ==========================================
needed_cols = [
    "Entrez Gene ID",
    "Official Symbol",
    "Sequence",
    "Refseq ID",
    "Post Translational Modification",
    "Residue",
    "Organism Name"
]

def load_dataset(folder_path):
    
    dfs = []

    for file in os.listdir(folder_path):

        if file.endswith(".txt"):

            file_path = os.path.join(folder_path, file)

            print(f"Reading: {file}")

            try:

                # Read messy TSV safely
                df = pd.read_csv(
                    file_path,
                    sep="\t",
                    engine="python",
                    # quoting=csv.QUOTE_NONE,
                    on_bad_lines="skip"
                )

                # Clean column names
                df.columns = df.columns.str.strip()

                # Keep only existing required columns
                existing_cols = [
                    col for col in needed_cols
                    if col in df.columns
                ]

                df = df[existing_cols]

                # Remove rows without protein names
                if "Official Symbol" in df.columns:
                    df = df.dropna(subset=["Official Symbol"])

                dfs.append(df)

                print(f"Loaded {file} -> {df.shape}")

            except Exception as e:

                print(f"Failed on {file}")
                print(e)

    # Combine all files
    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates
    if "Official Symbol" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(
            subset=["Official Symbol"]
        )

    return combined_df


# ==========================================
# LOAD BOTH DATASETS
# ==========================================
alz_df = load_dataset(alz_folder)
aut_df = load_dataset(aut_folder)

# ==========================================
# PRINT RESULTS
# ==========================================
print("\n========== ALZHEIMER ==========")
print(alz_df.head())
print(alz_df.shape)

print("\n========== AUTISM ==========")
print(aut_df.head())
print(aut_df.shape)

# ==========================================
# SAVE CLEAN DATA
# ==========================================
# alz_df.to_csv("alzheimers_clean.csv", index=False)
# aut_df.to_csv("autism_clean.csv", index=False)

print("\nSaved:")
print("alzheimers_clean.csv")
print("autism_clean.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  BUILD NETWORKX GRAPH
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(asd_df: pd.DataFrame, ad_df: pd.DataFrame) -> nx.Graph:
   
    G = nx.Graph()

    def add_edges(df: pd.DataFrame, disease_tag: str):
        for _, row in df.iterrows():
            a = row["Official Symbol"]
            b = row["Official Symbol"]
            pub = str(row.get("PublicGraphation Source", ""))
            exp = str(row.get("Experimental System", ""))

            # normalise edge key (alphabetical order)
            u, v = sorted([a, b])

            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
                G[u][v]["diseases"].add(disease_tag)
                if exp and exp != "nan":
                    G[u][v]["exp_systems"].add(exp)
            else:
                G.add_edge(
                    u, v,
                    weight=1,
                    diseases={disease_tag},
                    exp_systems={exp} if (exp and exp != "nan") else set(),
                )

            # update node disease label
            for node in (u, v):
                if node not in G.nodes:
                    G.add_node(node, disease=disease_tag)
                else:
                    cur = G.nodes[node].get("disease", disease_tag)
                    if cur != disease_tag:
                        G.nodes[node]["disease"] = "Both"
                    else:
                        G.nodes[node]["disease"] = disease_tag

    add_edges(asd_df, "ASD")
    add_edges(ad_df,  "AD")

    # convert sets to strings for serialisation
    for u, v, data in G.edges(data=True):
        data["diseases"]     = ";".join(sorted(data["diseases"]))
        data["exp_systems"]  = ";".join(sorted(data["exp_systems"]))

    return G


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ANALYSIS UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(G: nx.Graph, asd_df: pd.DataFrame, ad_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  NETWORK SUMMARY")
    print("=" * 60)
    print(f"  Nodes (proteins)  : {G.number_of_nodes():,}")
    print(f"  Edges (interactions): {G.number_of_edges():,}")
    print(f"  ASD interactions  : {len(asd_df):,}")
    print(f"  AD  interactions  : {len(ad_df):,}")

    # disease breakdown
    counts = Counter(nx.get_node_attributes(G, "disease").values())
    print(f"\n  Proteins in ASD only : {counts.get('ASD', 0):,}")
    print(f"  Proteins in AD  only : {counts.get('AD',  0):,}")
    print(f"  Proteins in BOTH     : {counts.get('Both', 0):,}")

    # top hubs
    degree = dict(G.degree())
    top10 = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top-10 hub proteins (by degree):")
    for rank, (gene, deg) in enumerate(top10, 1):
        disease = G.nodes[gene].get("disease", "?")
        print(f"    {rank:2d}. {gene:<12s}  degree={deg:4d}  ({disease})")
    print("=" * 60)


def get_top_subgraph(G: nx.Graph, n: int = 200) -> nx.Graph:
    """Return subgraph induced by the n highest-degree nodes."""
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:n]
    return G.subgraph([n for n, _ in top_nodes]).copy()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {"ASD": "#4A90D9", "AD": "#E8643A", "Both": "#7B2D8B"}
EDGE_COLORS = {"ASD": "#4A90D920", "AD": "#E8643A20", "Both": "#7B2D8B60"}


def visualise(G: nx.Graph, out_path: str = "network.png", n_nodes: int = 150):
    sub = get_top_subgraph(G, n=n_nodes)
    print(f"\n  Visualising top-{n_nodes} hub subgraph "
          f"({sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges) …")

    disease_map = nx.get_node_attributes(sub, "disease")
    node_colors = [COLORS.get(disease_map.get(nd, "ASD"), "#888") for nd in sub.nodes()]
    node_sizes  = [30 + 8 * sub.degree(nd) for nd in sub.nodes()]
    edge_weights= [sub[u][v]["weight"] for u, v in sub.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + 2.5 * w / max_w for w in edge_weights]
    edge_colors = [
        EDGE_COLORS.get(sub[u][v]["diseases"].split(";")[0], "#88888840")
        for u, v in sub.edges()
    ]

    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    pos = nx.spring_layout(sub, k=1.8 / (n_nodes ** 0.5), seed=42, iterations=60)

    nx.draw_networkx_edges(
        sub, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
    )
    nx.draw_networkx_nodes(
        sub, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        linewidths=0.4,
        edgecolors="#ffffff30",
    )

    # label only the very top hubs
    top_hub_names = {nd for nd, _ in sorted(sub.degree(), key=lambda x: x[1], reverse=True)[:30]}
    label_pos = {nd: (x, y + 0.015) for nd, (x, y) in pos.items() if nd in top_hub_names}
    nx.draw_networkx_labels(
        sub, label_pos,
        labels={nd: nd for nd in top_hub_names},
        ax=ax,
        font_size=6.5,
        font_color="white",
        font_family="monospace",
    )

    legend_handles = [
        mpatches.Patch(color=COLORS["ASD"],  label="ASD only"),
        mpatches.Patch(color=COLORS["AD"],   label="Alzheimer's only"),
        mpatches.Patch(color=COLORS["Both"], label="Shared (Both)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        framealpha=0.25,
        facecolor="#1a1a2e",
        edgecolor="#ffffff40",
        labelcolor="white",
        fontsize=11,
        title="Disease Association",
        title_fontsize=11,
    )

    ax.set_title(
        f"Protein Interaction Network — ASD & Alzheimer's Disease\n"
        f"Top {n_nodes} hub proteins  |  {sub.number_of_edges():,} edges shown",
        color="white", fontsize=14, pad=16,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5.  EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_graph(G: nx.Graph):
    """Save edge list and GraphML for downstream analysis."""
    # Edge list CSV
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "protein_A":   u,
            "protein_B":   v,
            "weight":      data["weight"],
            "diseases":    data["diseases"],
            "exp_systems": data["exp_systems"],
        })
    pd.DataFrame(rows).to_csv("network_edges.csv", index=False)
    print("  Saved → network_edges.csv")

    # Node attributes CSV
    node_rows = [
        {"protein": nd, "disease": attr.get("disease", ""), "degree": G.degree(nd)}
        for nd, attr in G.nodes(data=True)
    ]
    pd.DataFrame(node_rows).sort_values("degree", ascending=False).to_csv(
        "network_nodes.csv", index=False
    )
    print("  Saved → network_nodes.csv")

    # GraphML (compatible with Cytoscape, Gephi, etc.)
    nx.write_graphml(G, "network.graphml")
    print("  Saved → network.graphml")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[1/4] Loading datasets …")
    # asd_df = load_dataset("ASD", URLS["ASD"])
    # ad_df  = load_dataset("AD",  URLS["AD"])
    print(f"      ASD rows (human): {len(alz_df):,}")
    print(f"      AD  rows (human): {len(aut_df):,}")

    print("\n[2/4] Building NetworkX graph …")
    G = build_graph(alz_df, aut_df)
    print_summary(G, alz_df, aut_df)

    print("\n[3/4] Exporting files …")
    export_graph(G)

    print("\n[4/4] Generating visualisation …")
    visualise(G, out_path="network.png", n_nodes=150)

    print("\n✓  Done!  Files created:")
    print("     network.png      — visualisation (top-150 hubs)")
    print("     network_edges.csv — full edge list")
    print("     network_nodes.csv — node attributes & degree")
    print("     network.graphml   — importable in Cytoscape / Gephi")
    print()

    # ── quick access examples ──────────────────────────────────────────────
    # Largest connected component:
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    print(f"  Largest connected component: {lcc.number_of_nodes()} nodes, "
          f"{lcc.number_of_edges()} edges")

    # Shared proteins (appear in both diseases):
    shared = [n for n, d in nx.get_node_attributes(G, "disease").items() if d == "Both"]
    print(f"  Proteins shared between ASD & AD: {len(shared)}")
    if shared:
        top_shared = sorted(shared, key=lambda n: G.degree(n), reverse=True)[:10]
        print(f"  Top shared hubs: {', '.join(top_shared)}")