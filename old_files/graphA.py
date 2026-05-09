from preprocessing import *


G = nx.from_pandas_edgelist(
    df_filtered,
    source="protein1_name",
    target="protein2_name",
    edge_attr="combined_score"
)







largest_cc = max(nx.connected_components(G), key=len)
G_main = G.subgraph(largest_cc).copy()



# Top 10 hub proteins (highest degree)
top_hubs = sorted(G_main.degree(), key=lambda x: x[1], reverse=True)[:10]
for protein, degree in top_hubs:
    print(f"  {protein}  →  degree {degree}")


# sample_size = 500
# sampled_nodes = list(G_main.nodes())[:sample_size]
# G_sub = G_main.subgraph(sampled_nodes).copy()
sample_size = 200
nodes_sample = list(G.nodes())[:sample_size]
G_vis = G.subgraph(nodes_sample).copy()




# ── 2. Compute shortest path distances ────────────────────────────────────
lengths = dict(nx.all_pairs_shortest_path_length(G_vis))
nodes = list(G_vis.nodes())
N = len(nodes)

D = np.zeros((N, N))




for i, u in enumerate(nodes):
    for j, v in enumerate(nodes):
        D[i, j] = lengths[u].get(v, N)  # fallback for disconnected

hub_protein = top_hubs[0][0]                        # e.g. TP53
ego = nx.ego_graph(G_main, hub_protein, radius=3) 

if __name__ == "__main__":

    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")


    print(f"Largest component — nodes: {G_main.number_of_nodes():,}, edges: {G_main.number_of_edges():,}")
    print(f"Is connected: {nx.is_connected(G_main)}")
    print(f"Avg degree: {sum(d for _, d in G_main.degree()) / G_main.number_of_nodes():.1f}")


    print(f"Visualizing {G_vis.number_of_nodes()} nodes")




    # ── 3. Embed into 2D space (this creates your "space") ─────────────────────
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(D)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 10))

    # Draw edges (faint)
    for u, v in G_vis.edges():
        i, j = nodes.index(u), nodes.index(v)
        plt.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            alpha=0.2
        )

    # Draw nodes
    plt.scatter(coords[:, 0], coords[:, 1], s=20)

    # Optional: label a few important proteins
    for i, node in enumerate(nodes):
        plt.text(coords[i, 0], coords[i, 1], node, fontsize=8)

    plt.title("Topological Space of PPI Network")
    plt.axis("off")
    plt.show()
    # Option B — take ego network of a hub protein (more biologically meaningful)
    # 2-hop neighborhood
    print(f"Ego network size: {ego.number_of_nodes()} nodes")
