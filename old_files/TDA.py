from preprocessing import *
from graphA import *
def weighted_distance_matrix(G, weight_attr="weight", use_weights=True):
    """
    Convert STRING confidence scores → edge lengths.
    Higher confidence  →  shorter distance (proteins are 'closer').
    distance = 1 - (score / 1000)   so score=999 → dist=0.001
    """
    H = G.copy()

    if use_weights:
        for u, v, d in H.edges(data=True):
            raw = d.get(weight_attr, 700)          # fallback if missing
            d["length"] = 1.0 - (raw / 1000.0)    # ∈ (0, 1]

        lengths = dict(nx.all_pairs_dijkstra_path_length(H, weight="length"))
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(H))

    nodes = list(H.nodes())
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    # Use np.inf for disconnected, then impute with 2× max finite distance
    D = np.full((N, N), np.inf)
    for u, dists in lengths.items():
        for v, d in dists.items():
            D[idx[u]][idx[v]] = d

    finite_max = D[np.isfinite(D)].max()
    D[~np.isfinite(D)] = 2 * finite_max   # safer than N
    np.fill_diagonal(D, 0)

    return D, nodes



# ════════════════════════════════════════════════════════════════════════════════
# 3.  PERSISTENCE ENTROPY  (single-number complexity score)
# ════════════════════════════════════════════════════════════════════════════════

def persistence_entropy(dgm, remove_inf=True):
    if remove_inf:
        dgm = dgm[np.isfinite(dgm[:, 1])]
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return -np.sum(p * np.log(p + 1e-12))


# ════════════════════════════════════════════════════════════════════════════════
# 4.  BETTI NUMBER CURVE  (how many features alive at each filtration step)
# ════════════════════════════════════════════════════════════════════════════════

def betti_curve(dgm, t_values, remove_inf=True):
    if remove_inf:
        dgm = dgm[np.isfinite(dgm[:, 1])]
    counts = np.array([
        np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t))
        for t in t_values
    ])
    return counts




# ════════════════════════════════════════════════════════════════════════════════
# 5.  IDENTIFY SIGNIFICANT FEATURES  (long-lived = biologically meaningful)
# ════════════════════════════════════════════════════════════════════════════════

def significant_features(dgm, percentile=75, remove_inf=True):
    dgm_fin = dgm[np.isfinite(dgm[:, 1])] if remove_inf else dgm
    lifetimes = dgm_fin[:, 1] - dgm_fin[:, 0]
    threshold = np.percentile(lifetimes, percentile)
    mask = lifetimes >= threshold
    return dgm_fin[mask], lifetimes[mask], threshold



# ════════════════════════════════════════════════════════════════════════════════
# 6.  COMPARE TWO HUB PROTEINS  (bottleneck distance)
# ════════════════════════════════════════════════════════════════════════════════
# Replace HUB_A / HUB_B with actual node names from your graph, e.g. "TP53"

def compare_hubs(G_full, hub_a, hub_b, radius=3):
    ego_a = nx.ego_graph(G_full, hub_a, radius=radius)
    ego_b = nx.ego_graph(G_full, hub_b, radius=radius)

    Da, _ = weighted_distance_matrix(ego_a)
    Db, _ = weighted_distance_matrix(ego_b)

    dgm_a = ripser(Da, metric="precomputed", maxdim=1)["dgms"]
    dgm_b = ripser(Db, metric="precomputed", maxdim=1)["dgms"]

    # Bottleneck distance on H1
    h1_a = dgm_a[1][np.isfinite(dgm_a[1][:, 1])]
    h1_b = dgm_b[1][np.isfinite(dgm_b[1][:, 1])]
    dist  = bottleneck(h1_a, h1_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_diagrams(dgm_a, ax=axes[0], show=False)
    axes[0].set_title(f"H0/H1 — {hub_a}")
    plot_diagrams(dgm_b, ax=axes[1], show=False)
    axes[1].set_title(f"H0/H1 — {hub_b}")
    plt.suptitle(f"Bottleneck distance (H1): {dist:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig("tda_hub_comparison.png", dpi=150)
    plt.show()

    return dist


if __name__ == "__main__":


    D_weighted, nodes = weighted_distance_matrix(ego, use_weights=True)
    print(f"Distance matrix: {D_weighted.shape}  |  max dist: {D_weighted.max():.4f}")



    result = ripser(D_weighted, metric="precomputed", maxdim=1)
    dgms   = result["dgms"]
    H0, H1 = dgms[0], dgms[1]
    print(f"H0 features: {len(H0)}  |  H1 features: {len(H1)}  |")


    pe_H0 = persistence_entropy(H0)
    pe_H1 = persistence_entropy(H1)
    print(f"Persistence entropy — H0: {pe_H0:.4f}  |  H1: {pe_H1:.4f}")



    t = np.linspace(0, D_weighted.max() * 0.95, 300)
    b0 = betti_curve(H0, t)
    b1 = betti_curve(H1, t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t, b0, label="β₀ (components)", color="steelblue", lw=2)
    axes[0].plot(t, b1, label="β₁ (cycles)",     color="tomato",    lw=2)
    axes[0].set_xlabel("Filtration threshold")
    axes[0].set_ylabel("Betti number")
    axes[0].set_title("Betti Number Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Persistence diagram (cleaner version)
    plot_diagrams(dgms, ax=axes[1], show=False)
    axes[1].set_title("Persistence Diagram (H0, H1")
    plt.tight_layout()
    plt.savefig("tda_ppi_betti.png", dpi=150)
    plt.show()


    sig_H1, lives_H1, thresh_H1 = significant_features(H1, percentile=75)
    print(f"\nSignificant H1 cycles (top 25%):")
    print(f"  Threshold lifetime : {thresh_H1:.4f}")
    print(f"  Count              : {len(sig_H1)}")
    for birth, death in sig_H1:
        print(f"    born={birth:.4f}  died={death:.4f}  life={death-birth:.4f}")


    dist = compare_hubs(G, "TP53", "MYC")
    print(f"Topological distance between hubs: {dist:.4f}")



# ════════════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════════

    print("\n── TDA Summary ─────────────────────────────────────────────────────")
    print(f"  Nodes in ego graph      : {len(nodes)}")
    print(f"  H0 components           : {len(H0)}")
    print(f"  H1 cycles               : {len(H1)}")
    print(f"  Persistence entropy H0  : {pe_H0:.4f}")
    print(f"  Persistence entropy H1  : {pe_H1:.4f}")
    print(f"  Significant H1 cycles   : {len(sig_H1)}")
    print("───────────────────────")



