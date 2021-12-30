""" Класс графа """
import networkx as nx
from networkx.algorithms.shortest_paths import weighted
from networkx.readwrite import edgelist
import pandas as pd
import matplotlib.pyplot as plt

PSEUDO_INF = 999

class Graph:
    StartVertices = [] # Стратовые вершины
    FinishVertices = [] # Финишные вершины
    Weights = [] # Веса существующих рёбер
    WeightsMatrix = [] # Матрица весов, для удобного обхода 
    VertexCount = 0
    EdgesCount = 0
    
    def __init__(self): # По умолчанию граф создаётся из файла graph.txt
        with open("graph.txt", 'r') as f:
            txtgraph = f.readlines() # Список строк файла
            
            self.VertexCount = int(txtgraph[0].split()[0])
            self.EdgesCount = int(txtgraph[0].split()[1])
            
            for i in range(self.EdgesCount): # Формируем списки вершин и весов
                self.StartVertices.append(int(txtgraph[1+self.VertexCount+i].split()[0]))
                self.FinishVertices.append(int(txtgraph[1+self.VertexCount+i].split()[1]))
                self.Weights.append(int(txtgraph[1+self.VertexCount+i].split()[2]))
            
            for i in range(self.VertexCount): # По умолчанию нет ребер
                temp = [PSEUDO_INF]*self.VertexCount
                self.WeightsMatrix.append(temp)
                            
            for i in range(self.EdgesCount): # Добавляем веса ребер
                self.WeightsMatrix[self.StartVertices[i]][self.FinishVertices[i]] = self.Weights[i]
                
            for i in range(self.VertexCount): # Из самой в себя 0
                self.WeightsMatrix[i][i] = 0
                    
            for i in range(self.VertexCount):
                print(self.WeightsMatrix[i])
            
    def ShowGraph(self):
        DF = pd.DataFrame({'S': self.StartVertices, 'F': self.FinishVertices, "weight": self.Weights}) # Формируем dataframe рёбер
        print(DF) # Лог вывод
        
        G = nx.from_pandas_edgelist(df=DF, source='S', target='F', edge_attr="weight", create_using=nx.DiGraph()) # Присоединяем dataframe к графу

        # Рисуем граф
        pos = nx.shell_layout(G)
        nx.draw(G, pos, with_labels=True)
        labels = {e: G.edges[e]['weight'] for e in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()