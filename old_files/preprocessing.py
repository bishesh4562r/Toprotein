import pandas as pd
import networkx as nx
from ripser import ripser
from persim import plot_diagrams, bottleneck
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
import warnings
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")

df = pd.read_csv(
    "data/9606.protein.physical.links.v12.0.txt.gz",
    sep=" ",
    header=0
)

print(df.head())
print(f"Total edges: {len(df):,}")

# ── 2. Filter by confidence ──────────────────────────────────────────────────
THRESHOLD = 800
df_filtered = df[df["combined_score"] >= THRESHOLD].copy()

print(f"Edges after threshold {THRESHOLD}: {len(df_filtered):,}")

# ── 3. Load alias file ───────────────────────────────────────────────────────
aliases = pd.read_csv(
    "data/9606.protein.aliases.v12.0.txt.gz",
    sep="\t"
)

# Keep only human-readable gene names
preferred_sources = ["Ensembl_HGNC_symbol", "UniProtKB-ID"]

gene_aliases = aliases[
    aliases["source"].isin(preferred_sources)
].drop_duplicates("#string_protein_id")

# Build mapping dictionary
alias_dict = dict(zip(
    gene_aliases["#string_protein_id"],
    gene_aliases["alias"]
))

# ── 4. Map protein IDs → gene names ──────────────────────────────────────────
def map_name(p):
    return alias_dict.get(p, p)  # fallback if missing

df_filtered["protein1_name"] = df_filtered["protein1"].apply(map_name)
df_filtered["protein2_name"] = df_filtered["protein2"].apply(map_name)

# ── 5. Build graph with readable names ───────────────────────────────────────




# G_tda = ego 

# lengths = dict(nx.all_pairs_shortest_path_length(G_tda))
# nodes = list(G_tda.nodes())
# N = len(nodes)

# D = np.zeros((N, N))
# for i, u in enumerate(nodes):
#     for j, v in enumerate(nodes):
#         D[i][j] = lengths[u].get(v, N)   # N as proxy for "disconnected"

# print(f"Distance matrix shape: {D.shape}")



# diagrams = ripser(D, metric="precomputed", maxdim=1, distance_matrix=True)["dgms"]

# plot_diagrams(diagrams, show=True)
# plt.title("Persistence diagram — PPI subgraph")
# plt.show()

