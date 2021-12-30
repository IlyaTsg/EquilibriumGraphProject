import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph() # Создаём пустой граф

G.add_nodes_from(['A', 'B', 'C', 'D']) # Добавим вершины из списка

G.add_edges_from([ ('A', 'B', {"weight": 16}), ('B', 'D', {"weight": 16}), ('C', 'A', {"weight": 6}), ('B', 'C', {"weight": 27})]) # С весами ребёр

#G.edges() # Получение ребер
#G.edges(data = True) # Получение ребер и весов

pos = nx.spring_layout(G, k=10)
nx.draw(G, pos, with_labels=True)
labels = {e: G.edges[e]['weight'] for e in G.edges} # Создаём словарь из весов ребёр, чтобы передать его для рисования
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()