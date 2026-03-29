def correlate_centrality(df, label):
    measures = ["Norm In-Degree", "Norm Out-Degree", "Norm Total Degree",
                "Norm Betweenness", "Norm Closeness", "Norm Eigenvector"]
    corr = df[measures].corr()
    return corr
