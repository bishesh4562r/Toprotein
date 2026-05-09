import requests
import pandas as pd
import networkx as nx
from io import StringIO
from time import sleep
import os
# ======================================================
# SETTINGS
# ======================================================

DATASETS = {
    "cancer": 'annot:"dataset:cancer"',

    # healthy human interactions
    "healthy": 'species:human'
}
BATCH_SIZE = 500          # entries per request
MAX_RECORDS = 10000
       # total records to fetch

# ======================================================
# GRAPH
# ======================================================

G = nx.Graph()

# ======================================================
# HELPERS
# ======================================================

def extract_id(x):
    """
    Convert:
    uniprotkb:P12345 -> P12345
    """

    if pd.isna(x):
        return None

    x = str(x)

    if ":" in x:
        return x.split(":")[1]

    return x


def extract_confidence(x):
    """
    Extract:
    intact-miscore:0.87 -> 0.87
    """

    if pd.isna(x):
        return None

    x = str(x)

    try:
        if ":" in x:
            return float(x.split(":")[-1])
    except:
        pass

    return None

# ======================================================
def build_network(query_name, query_string):
    
    print("\n" + "=" * 60)
    print(f"BUILDING {query_name.upper()} NETWORK")
    print("=" * 60)

    G = nx.Graph()

    base_url = (
        "https://www.ebi.ac.uk/Tools/webservices/psicquic/"
        "intact/webservices/current/search/query/"
    )

    for start in range(0, MAX_RECORDS, BATCH_SIZE):

        print(f"\nFetching records {start} -> {start + BATCH_SIZE}")
        url = (
            f"{base_url}{query_string}"
            f"?format=tab25"
            f"&firstResult={start}"
            f"&maxResults={BATCH_SIZE}"
        )

        response = requests.get(url)

        if response.status_code != 200:
            print("Request failed.")
            break

        text = response.text.strip()

        if not text:
            print("No more records.")
            break

        df = pd.read_csv(
            StringIO(text),
            sep="\t",
            header=None,
            low_memory=False
        )

        print("Fetched:", len(df))

        df = df[[0, 1, 14]].copy()

        df.columns = [
            "molecule_A",
            "molecule_B",
            "confidence"
        ]

        df["molecule_A"] = df["molecule_A"].apply(extract_id)
        df["molecule_B"] = df["molecule_B"].apply(extract_id)

        df["confidence"] = df["confidence"].apply(
            extract_confidence
        )

        df = df.dropna(
            subset=["molecule_A", "molecule_B"]
        )

        for _, row in df.iterrows():

            a = row["molecule_A"]
            b = row["molecule_B"]
            conf = row["confidence"]

            if G.has_edge(a, b):

                old_conf = G[a][b].get("confidence")

                if old_conf is None:
                    old_conf = 0

                if conf is not None and conf > old_conf:
                    G[a][b]["confidence"] = conf
                    G[a][b]["weight"] = conf

            else:

                G.add_edge(
                    a,
                    b,
                    confidence=conf,
                    weight=conf if conf is not None else 1.0
                )

        print(
            f"Graph now has "
            f"{G.number_of_nodes()} nodes and "
            f"{G.number_of_edges()} edges"
        )

        sleep(0.5)

    print("\n==============================")
    print(f"FINAL {query_name.upper()} GRAPH")
    print("==============================")

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    output_dir = f"{query_name.capitalize()}Graph"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{query_name}_network.gml"

    nx.write_gml(G, output_file)

    print(f"\nSaved: {output_file}")

    return G

# ======================================================
# FETCH LOOP
# ======================================================

# base_url = (
#     "https://www.ebi.ac.uk/Tools/webservices/psicquic/"
#     "intact/webservices/current/search/query/"
# )

# for start in range(0, MAX_RECORDS, BATCH_SIZE):

#     print(f"\nFetching records {start} -> {start + BATCH_SIZE}")

#     url = (
#         f"{base_url}{QUERY}"
#         f"?format=tab25"
#         f"&firstResult={start}"
#         f"&maxResults={BATCH_SIZE}"
#     )

#     response = requests.get(url)

#     if response.status_code != 200:
#         print("Request failed.")
#         break

#     text = response.text.strip()

#     # stop if no more data
#     if not text:
#         print("No more records.")
#         break

#     # ==================================================
#     # READ MITAB
#     # ==================================================

#     df = pd.read_csv(
#         StringIO(text),
#         sep="\t",
#         header=None,
#         low_memory=False
#     )

#     print("Fetched:", len(df))

#     # Need:
#     # 0  -> interactor A
#     # 1  -> interactor B
#     # 14 -> confidence score

#     df = df[[0, 1, 14]].copy()

#     df.columns = [
#         "molecule_A",
#         "molecule_B",
#         "confidence"
#     ]

#     # ==================================================
#     # CLEAN
#     # ==================================================

#     df["molecule_A"] = df["molecule_A"].apply(extract_id)
#     df["molecule_B"] = df["molecule_B"].apply(extract_id)

#     df["confidence"] = df["confidence"].apply(
#         extract_confidence
#     )

#     df = df.dropna(
#         subset=["molecule_A", "molecule_B"]
#     )

#     # ==================================================
#     # BUILD GRAPH
#     # ==================================================

#     for _, row in df.iterrows():

#         a = row["molecule_A"]
#         b = row["molecule_B"]
#         conf = row["confidence"]

#         # if edge already exists, keep maximum confidence
#         if G.has_edge(a, b):

#             old_conf = G[a][b].get("confidence")

#             if old_conf is None:
#                 old_conf = 0

#             if conf is not None and conf > old_conf:
#                 G[a][b]["confidence"] = conf
#                 G[a][b]["weight"] = conf

#         else:

#             G.add_edge(
#                 a,
#                 b,
#                 confidence=conf,
#                 weight=conf if conf is not None else 1.0
#             )

#     print(
#         f"Graph now has "
#         f"{G.number_of_nodes()} nodes and "
#         f"{G.number_of_edges()} edges"
#     )

#     # be polite to server
#     sleep(0.5)

# # ======================================================
# # FINAL STATS
# # ======================================================

# print("\n==============================")
# print("FINAL GRAPH")
# print("==============================")

# print("Nodes:", G.number_of_nodes())
# print("Edges:", G.number_of_edges())

# # ======================================================
# # SAVE
# # ======================================================

# nx.write_gml(G, "CancerGraph/cancer_network.gml")

# print("\nSaved: cancer_network.gml")

networks = {}

for dataset_name, dataset_query in DATASETS.items():

    networks[dataset_name] = build_network(
        dataset_name,
        dataset_query
    )

print("\nAll datasets processed successfully.")