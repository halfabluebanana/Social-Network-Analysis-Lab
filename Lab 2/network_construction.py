import networkx as nx
from collections import defaultdict


def build_network(edges, node_data, label):
    """Build directed weighted NetworkX graph."""
    G = nx.DiGraph()

    edge_weights = defaultdict(int)
    for e in edges:
        edge_weights[(e["source"], e["target"])] += 1

    for (src, tgt), weight in edge_weights.items():
        G.add_edge(src, tgt, weight=weight)

    for node in G.nodes():
        nd = node_data.get(node, {})
        G.nodes[node]["comment_count"] = nd.get("comment_count", 0)
        cc = nd.get("comment_count", 1)
        G.nodes[node]["avg_score"] = nd.get("total_score", 0) / max(cc, 1)
        G.nodes[node]["network"] = label

    print(f"\nNetwork '{label}'")
    print(f"   Nodes       : {G.number_of_nodes()}")
    print(f"   Edges       : {G.number_of_edges()}")
    print(f"   Density     : {nx.density(G):.5f}")
    try:
        print(f"   Reciprocity : {nx.reciprocity(G):.4f}")
    except Exception:
        print(f"   Reciprocity : N/A")

    return G
