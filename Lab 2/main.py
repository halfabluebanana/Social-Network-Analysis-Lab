from data_collection import collect_network, collect_ira_network
from network_construction import build_network
from topography import describe_network
from centrality import compute_centrality, highlight_central_nodes
from correlations import correlate_centrality
from visualizations import plot_all
from alternate_specs import alternate_specs
from config import OUTPUT_DIR


def main():
    print("Dead Internet Theory — Reddit Network Analysis")
    print("=" * 50)

    # Collect
    edges_bots,   nodes_bots   = collect_network("SubSimulatorGPT2", n_posts=15, label="Bots")
    edges_humans, nodes_humans = collect_network("changemyview",      n_posts=15, label="Humans")
    edges_ira,    nodes_ira    = collect_ira_network()

    # Build
    G_bots   = build_network(edges_bots,   nodes_bots,   "r/SubSimulatorGPT2 (Bots)")
    G_humans = build_network(edges_humans, nodes_humans, "r/changemyview (Humans)")
    G_ira    = build_network(edges_ira,    nodes_ira,    "IRA Bots")

    describe_network(G_bots,   "r/SubSimulatorGPT2 (Bots)")
    describe_network(G_humans, "r/changemyview (Humans)")
    describe_network(G_ira,    "IRA Bots")

    # Centrality
    df_bots   = compute_centrality(G_bots,   "Bots")
    df_humans = compute_centrality(G_humans, "Humans")
    df_ira    = compute_centrality(G_ira,    "IRA")

    highlight_central_nodes(df_bots,   "Bots")
    highlight_central_nodes(df_humans, "Humans")
    highlight_central_nodes(df_ira,    "IRA")

    # Correlations
    corr_bots   = correlate_centrality(df_bots,   "Bots")
    corr_humans = correlate_centrality(df_humans, "Humans")
    corr_ira    = correlate_centrality(df_ira,    "IRA")

    # Visualisations
    print("\nGenerating visualisations...")
    plot_all(df_bots, df_humans, G_bots, G_humans, corr_bots, corr_humans,
             df_ira=df_ira, G_ira=G_ira, corr_ira=corr_ira)

    # Alternate specs
    alternate_specs(G_bots, G_humans, df_bots, df_humans, G_ira=G_ira, df_ira=df_ira,
                    edges_bots=edges_bots, edges_humans=edges_humans, edges_ira=edges_ira)

    # Save CSVs
    df_bots.to_csv(f"{OUTPUT_DIR}/centrality_bots.csv")
    df_humans.to_csv(f"{OUTPUT_DIR}/centrality_humans.csv")
    df_ira.to_csv(f"{OUTPUT_DIR}/centrality_ira.csv")
    corr_bots.to_csv(f"{OUTPUT_DIR}/corr_bots.csv")
    corr_humans.to_csv(f"{OUTPUT_DIR}/corr_humans.csv")
    corr_ira.to_csv(f"{OUTPUT_DIR}/corr_ira.csv")

    print(f"\nAll done. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
