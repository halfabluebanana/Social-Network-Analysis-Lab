import networkx as nx
import numpy as np


def describe_network(G, label):
    print(f"\n{'='*50}")
    print(f"TOPOGRAPHY — {label}")
    print(f"{'='*50}")

    degrees  = dict(G.degree())
    in_deg   = dict(G.in_degree())
    out_deg  = dict(G.out_degree())

    print(f"Nodes              : {G.number_of_nodes()}")
    print(f"Edges              : {G.number_of_edges()}")
    print(f"Density            : {nx.density(G):.6f}")
    try:
        print(f"Reciprocity        : {nx.reciprocity(G):.4f}")
    except Exception:
        pass
    print(f"Avg total degree   : {np.mean(list(degrees.values())):.2f}")
    print(f"Avg in-degree      : {np.mean(list(in_deg.values())):.2f}")
    print(f"Avg out-degree     : {np.mean(list(out_deg.values())):.2f}")
    print(f"Max degree node    : {max(degrees, key=degrees.get)} "
          f"(degree={max(degrees.values())})")

    wccs = list(nx.weakly_connected_components(G))
    print(f"Weakly conn comps  : {len(wccs)}")
    print(f"Largest component  : {max(len(c) for c in wccs)} nodes "
          f"({100*max(len(c) for c in wccs)/G.number_of_nodes():.1f}%)")
