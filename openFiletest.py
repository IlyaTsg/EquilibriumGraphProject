import networkx as nx
from networkx.readwrite import edgelist
import pandas as pd
import matplotlib.pyplot as plt

with open("graph.txt", 'r') as f:
    txtgraph = f.readlines() # Список строк файла

    VertexCount = int(txtgraph[0].split()[0])
    EdgesCount = int(txtgraph[0].split()[1])

    StartVertices = []
    FinishVertices = []
    Weights = []
    for i in range(EdgesCount): # Формируем списки вершин и весов
        StartVertices.append(txtgraph[1+VertexCount+i].split()[0])
        FinishVertices.append(txtgraph[1+VertexCount+i].split()[1])
        Weights.append(txtgraph[1+VertexCount+i].split()[2])

    DF = pd.DataFrame({'S': StartVertices, 'F': FinishVertices, "weight": Weights}) # Формируем dataframe рёбер
    print(DF)
    
    G = nx.from_pandas_edgelist(df=DF, source='S', target='F', edge_attr="weight", create_using=nx.DiGraph()) # Присоединяем dataframe к графу

    # Рисуем граф
    pos = nx.shell_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = {e: G.edges[e]['weight'] for e in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


