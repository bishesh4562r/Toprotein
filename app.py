import streamlit as st
import json
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import tempfile
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cancer vs Healthy PPI Analysis",
    layout="wide"
)

st.title("🧬 Cancer vs Healthy PPI Comparative Analysis")
st.markdown("Interactive Spectral + TDA Analysis Dashboard")

# =========================================================
# FILE PATHS
# =========================================================

FILES = {
    "cancer_graph": "CancerGraph/cancer_network.gml",
    "healthy_graph": "HealthyGraph/healthy_network.gml",

    "cancer_spectral": "cancer_spectral_properties.json",
    "healthy_spectral": "healthy_spectral_properties.json",
    "comparative_spectral": "comparative_spectral_analysis.json",

    "cancer_tda": "cancer_tda_results.json",
    "healthy_tda": "healthy_tda_results.json",
    "comparative_tda": "comparative_tda_analysis.json"
}

# =========================================================
# HELPERS
# =========================================================

@st.cache_data

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource

def load_graph(path):
    return nx.read_gml(path)


# =========================================================
# LOAD DATA
# =========================================================

cancer_graph = load_graph(FILES["cancer_graph"])
healthy_graph = load_graph(FILES["healthy_graph"])

cancer_spectral = load_json(FILES["cancer_spectral"])
healthy_spectral = load_json(FILES["healthy_spectral"])
comparative_spectral = load_json(FILES["comparative_spectral"])

cancer_tda = load_json(FILES["cancer_tda"])
healthy_tda = load_json(FILES["healthy_tda"])
comparative_tda = load_json(FILES["comparative_tda"])

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Graph Statistics",
        "Spectral Analysis",
        "TDA Analysis",
        "Interactive Graph Explorer",
        "Comparative Metrics"
    ]
)

# =========================================================
# OVERVIEW
# =========================================================

if section == "Overview":

    st.header("📊 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cancer Network")
        st.metric("Nodes", cancer_graph.number_of_nodes())
        st.metric("Edges", cancer_graph.number_of_edges())

    with col2:
        st.subheader("Healthy Network")
        st.metric("Nodes", healthy_graph.number_of_nodes())
        st.metric("Edges", healthy_graph.number_of_edges())

    st.markdown("---")

    st.subheader("Key Scientific Interpretation")

    st.markdown(
        """
        - Spectral Radius → network influence and diffusion power
        - Spectral Gap → modularity and synchronization
        - Algebraic Connectivity → robustness and resilience
        - Betti-0 → connected components
        - Betti-1 → loops/cycles/topological complexity
        - Persistence Entropy → structural disorder/complexity
        """
    )

# =========================================================
# GRAPH STATS
# =========================================================

elif section == "Graph Statistics":

    st.header("📈 Graph Statistics")

    stats_df = pd.DataFrame({
        "Metric": [
            "Nodes",
            "Edges",
            "Average Degree",
            "Density"
        ],

        "Cancer": [
            cancer_graph.number_of_nodes(),
            cancer_graph.number_of_edges(),
            round(np.mean([d for n, d in cancer_graph.degree()]), 4),
            round(nx.density(cancer_graph), 6)
        ],

        "Healthy": [
            healthy_graph.number_of_nodes(),
            healthy_graph.number_of_edges(),
            round(np.mean([d for n, d in healthy_graph.degree()]), 4),
            round(nx.density(healthy_graph), 6)
        ]
    })

    st.dataframe(stats_df, use_container_width=True)

    fig = px.bar(
        stats_df,
        x="Metric",
        y=["Cancer", "Healthy"],
        barmode="group",
        title="Cancer vs Healthy Graph Statistics"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SPECTRAL ANALYSIS
# =========================================================

elif section == "Spectral Analysis":

    st.header("🧠 Spectral Analysis")

    cancer_props = cancer_spectral["spectral_properties"]
    healthy_props = healthy_spectral["spectral_properties"]

    spectral_df = pd.DataFrame({
        "Property": [
            "Spectral Radius",
            "Largest Eigenvalue",
            "Spectral Gap",
            "Algebraic Connectivity"
        ],

        "Cancer": [
            cancer_props["spectral_radius"],
            cancer_props["largest_adjacency_eigenvalue"],
            cancer_props["spectral_gap"],
            cancer_props["algebraic_connectivity"]
        ],

        "Healthy": [
            healthy_props["spectral_radius"],
            healthy_props["largest_adjacency_eigenvalue"],
            healthy_props["spectral_gap"],
            healthy_props["algebraic_connectivity"]
        ]
    })

    st.dataframe(spectral_df, use_container_width=True)

    fig = px.bar(
        spectral_df,
        x="Property",
        y=["Cancer", "Healthy"],
        barmode="group",
        title="Spectral Property Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Adjacency Eigenvalues")

    eig_df = pd.DataFrame({
        "Index": list(range(20)),

        "Cancer": cancer_spectral[
            "top_20_adjacency_eigenvalues"
        ],

        "Healthy": healthy_spectral[
            "top_20_adjacency_eigenvalues"
        ]
    })

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=eig_df["Index"],
        y=eig_df["Cancer"],
        mode="lines+markers",
        name="Cancer"
    ))

    fig2.add_trace(go.Scatter(
        x=eig_df["Index"],
        y=eig_df["Healthy"],
        mode="lines+markers",
        name="Healthy"
    ))

    fig2.update_layout(
        title="Top 20 Adjacency Eigenvalues",
        xaxis_title="Eigenvalue Rank",
        yaxis_title="Eigenvalue"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# TDA ANALYSIS
# =========================================================

elif section == "TDA Analysis":

    st.header("🔺 Topological Data Analysis")

    cancer_entropy = cancer_tda["persistence_entropy"]
    healthy_entropy = healthy_tda["persistence_entropy"]

    tda_df = pd.DataFrame({
        "Metric": [
            "H0 Count",
            "H1 Count",
            "Persistence Entropy H0",
            "Persistence Entropy H1"
        ],

        "Cancer": [
            cancer_tda["betti_numbers"]["H0_count"],
            cancer_tda["betti_numbers"]["H1_count"],
            cancer_entropy["H0"],
            cancer_entropy["H1"]
        ],

        "Healthy": [
            healthy_tda["betti_numbers"]["H0_count"],
            healthy_tda["betti_numbers"]["H1_count"],
            healthy_entropy["H0"],
            healthy_entropy["H1"]
        ]
    })

    st.dataframe(tda_df, use_container_width=True)

    fig = px.bar(
        tda_df,
        x="Metric",
        y=["Cancer", "Healthy"],
        barmode="group",
        title="Topological Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Betti Curves")

    cancer_thresholds = cancer_tda[
        "betti_curves"
    ]["thresholds"]

    healthy_thresholds = healthy_tda[
        "betti_curves"
    ]["thresholds"]

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=cancer_thresholds,
        y=cancer_tda["betti_curves"]["beta_1"],
        mode="lines",
        name="Cancer β1"
    ))

    fig2.add_trace(go.Scatter(
        x=healthy_thresholds,
        y=healthy_tda["betti_curves"]["beta_1"],
        mode="lines",
        name="Healthy β1"
    ))

    fig2.update_layout(
        title="Betti-1 Curves",
        xaxis_title="Filtration Threshold",
        yaxis_title="β1"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# INTERACTIVE GRAPH EXPLORER
# =========================================================

elif section == "Interactive Graph Explorer":

    st.header("🕸 Interactive Graph Explorer")

    graph_choice = st.selectbox(
        "Choose Network",
        ["Cancer", "Healthy"]
    )

    if graph_choice == "Cancer":
        G = cancer_graph
    else:
        G = healthy_graph

    st.write(
        f"Displaying first 300 nodes for performance."
    )

    nodes_subset = list(G.nodes())[:300]

    subG = G.subgraph(nodes_subset)

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#111111",
        font_color="white"
    )

    for node in subG.nodes():
        net.add_node(node, label=str(node))

    for u, v in subG.edges():
        net.add_edge(u, v)

    net.repulsion(
        node_distance=120,
        central_gravity=0.2,
        spring_length=100,
        spring_strength=0.05,
        damping=0.09
    )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".html"
    )

    net.save_graph(temp_file.name)

    with open(temp_file.name, "r", encoding="utf-8") as f:
        html_data = f.read()

    st.components.v1.html(html_data, height=700)

# =========================================================
# COMPARATIVE METRICS
# =========================================================

elif section == "Comparative Metrics":

    st.header("⚖ Comparative Metrics")

    st.subheader("Spectral Differences")

    spectral_diff = comparative_spectral[
        "comparative_metrics"
    ]

    spectral_diff_df = pd.DataFrame({
        "Metric": list(spectral_diff.keys()),
        "Difference": list(spectral_diff.values())
    })

    st.dataframe(
        spectral_diff_df,
        use_container_width=True
    )

    fig1 = px.bar(
        spectral_diff_df,
        x="Metric",
        y="Difference",
        title="Spectral Differences"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Topological Differences")

    topo_diff = comparative_tda[
        "comparative_topology"
    ]

    topo_diff_df = pd.DataFrame({
        "Metric": list(topo_diff.keys()),
        "Difference": list(topo_diff.values())
    })

    st.dataframe(
        topo_diff_df,
        use_container_width=True
    )

    fig2 = px.bar(
        topo_diff_df,
        x="Metric",
        y="Difference",
        title="Topological Differences"
    )

    st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    ### Biological Interpretation

    - Higher spectral radius → stronger information propagation
    - Higher algebraic connectivity → more robust network
    - Higher H1 cycles → richer topological loops
    - Higher persistence entropy → increased structural complexity

    Cancer PPI networks often exhibit:

    - rewired signaling
    - abnormal modularity
    - higher heterogeneity
    - altered topological persistence
    """
)