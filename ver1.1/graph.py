""" Класс графа """

# Правильный ввод зависимости: 10x + 12 или 0x + 12 или 3x
FLOW = 6 # Общий поток

from os import PathLike
import networkx as nx
from networkx.algorithms.shortest_paths import weighted
from networkx.readwrite import edgelist
from numpy.lib.function_base import append
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import solve

class Graph:
    StartVertices = [] # Стратовые вершины
    FinishVertices = [] # Финишные вершины
    Matrix = [] # Матрица смежности
    Puths = [] # Список всех путей от истока до стока
    PuthsMatrix = [] # Матрица путей для рёбер
    EdgesExpressionList = [] # Список функций для рёбер
    PuthsCount = 0
    VertexCount = 0
    EdgesCount = 0
    
    def __init__(self): # По умолчанию граф создаётся из файла graph.txt
        with open("graph.dot", 'r') as f:
            txtgraph = f.readlines() # Список строк файла

            verticeslist = []

            for i in txtgraph:
                if i != '}' and i != "digraph {\n":
                    self.EdgesCount += 1
                    temp_S = int(i.split()[0])-1
                    self.StartVertices.append(temp_S)
                    temp = i.split()[2]
                    temp_F = int(temp[0:1])-1
                    self.FinishVertices.append(temp_F)
                    if(temp_S not in verticeslist):
                        verticeslist.append(temp_S)
                        self.VertexCount += 1
                    if(temp_F not in verticeslist):
                        verticeslist.append(temp_F)
                        self.VertexCount += 1
                        
            for i in range(self.VertexCount): # По умолчанию нет ребер
                temp = [0]*self.VertexCount
                self.Matrix.append(temp)
                            
            for i in range(self.EdgesCount): # Добавляем 1, если есть ребро
                self.Matrix[self.StartVertices[i]][self.FinishVertices[i]] = 1
            
    def ShowGraph(self):
        DF = pd.DataFrame({'S': self.StartVertices, 'F': self.FinishVertices}) # Формируем dataframe рёбер
        print(DF) # Лог вывод
        
        G = nx.from_pandas_edgelist(df=DF, source='S', target='F', create_using=nx.DiGraph()) # Присоединяем dataframe к графу

        # Рисуем граф
        pos = nx.shell_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.show()
        
    def GetPuths(self, matrix, start, finish, visited: list):
        visited.append(start)
    
        for i in range(self.VertexCount):
            if matrix[start][i] == 1 and i not in visited:
                if i == finish:
                    visited.append(finish)
                    self.Puths.append(visited)
                    self.PuthsCount += 1
                else:
                    self.GetPuths(matrix, i, finish, visited.copy())
    
    def GetPuthMatrix(self):
        for i in range(self.EdgesCount):
            self.PuthsMatrix.append([0]*self.PuthsCount)
            
        for i in range(self.PuthsCount):
            for j in range(len(self.Puths[i])-1):
                for k in range(len(self.StartVertices)):
                    if self.StartVertices[k] == self.Puths[i][j] and self.FinishVertices[k] == self.Puths[i][j+1]:
                        self.PuthsMatrix[k][i] = 1
            
    """ Класс функции ребра"""
    class Edge:
        coef = np.array(0) # Коэффициенты при x
        free = np.array(0) # Свободные члены

        def __init__(self, expression: str):
            wx = expression.split()[0]
            wx = wx[0:len(wx)-1]
            self.coef = np.delete(self.coef, 0)
            self.free = np.delete(self.free, 0)
            if(len(wx) == 0):
                self.coef = np.append(self.coef, 1)
            else:
                self.coef = np.append(self.coef, int(wx))
            if(len(wx)+1 != len(expression)):
                if(expression.split()[1] == '-'):
                    self.free = np.append(self.free, int(expression.split()[2])*(-1))
                else:
                    self.free = np.append(self.free, int(expression.split()[2]))
            else:
                self.free = np.append(self.free, 0)
            
    # Составление и решение уравнения для двух путей
    def EqualTime(self, EdgesExpressionList, Paths, NpathA, NpathB):
        coefs = np.array([0]*self.PuthsCount)
        frees = np.array([0])
        
        for i in range(self.EdgesCount):
            if Paths[i][NpathA]: # Если ребро учавствует в пути
                for j in range(self.PuthsCount): # Оставльные потоки тоже нужно учитывать
                    if Paths[i][j]:
                        coefs[j] += EdgesExpressionList[i].coef
                        frees[0] -= EdgesExpressionList[i].free
            if Paths[i][NpathB]:
                for j in range(self.PuthsCount): # Минусуем все переменные, т.к перенос
                    if Paths[i][j]:
                        coefs[j] -= EdgesExpressionList[i].coef
                        frees[0] += EdgesExpressionList[i].free
        
        print("DEBUG func(EqualTime)", coefs)
        print("DEBUG func(EqualTime)", frees)
        
        result = []
        result.append(coefs)
        result.append(frees)
        return result
    
    def PrepareExpressionList(self, text: str):
        j = 0
        for i in range(self.EdgesCount):
            exp = ""
            while  j != len(text) and text[j] != "\n":    
                exp += text[j]
                j += 1
            j += 1
            self.EdgesExpressionList.append(self.Edge(exp))
            
    def BalanceEquation(self, InBalance: list): # Поиск равновесного распределения
        coefs = np.array([], dtype="int32")
        frees = np.array([], dtype="int32")
        EquationCount = 0
        
        for i in range(self.PuthsCount):
            if InBalance[i] == 1:
                for j in range(i+1, self.PuthsCount):
                    if InBalance[j] == 1:
                        result = self.EqualTime(self.EdgesExpressionList, self.PuthsMatrix, i, j)
                        coefs = np.append(coefs, result[0])
                        frees = np.append(frees, result[1])
                        EquationCount += 1
                break;
                
        coefs = np.append(coefs, InBalance)
        frees = np.append(frees, FLOW)
        EquationCount += 1
        
        for j in range(len(InBalance)):
            temp = [0]*self.PuthsCount
            if InBalance[j] == 0:
                temp[j] = 1
                coefs = np.append(coefs, np.array(temp))
                frees = np.append(frees, 0)
        
        coefs = np.reshape(coefs, (self.PuthsCount, self.PuthsCount))
        frees = np.reshape(frees, (self.PuthsCount, 1))
        
        res = solve(coefs, frees)
        
        IsFloat = False
        for i in range(self.PuthsCount):
            tod = str(res[i]).index('.')
            if len(str(res[i])[tod:]) != 2:
                IsFloat = True
        
        if IsFloat:
            return [0]*self.PuthsCount
        return res
    
    def PathTime(self, PuthNum, Balance: list):
        res = 0
        for i in range(self.EdgesCount):
            if self.PuthsMatrix[i][PuthNum] == 1:
                for j in range(self.PuthsCount):
                    if self.PuthsMatrix[i][j] == 1:
                        res += self.EdgesExpressionList[i].coef*Balance[j] + self.EdgesExpressionList[i].free
        return res
    
    def FindBalance(self):
        AllBalances = []
        
        # Находим все возможные распределения
        InBalance = []
        for i in range(1, 2**self.PuthsCount):
            temp = [0]*(len(bin(2**self.PuthsCount-1)[2:]))
            BinInBalance = bin(i)[2:].zfill(len(bin(2**self.PuthsCount-1)[2:]))
            for j in range(len(bin(2**self.PuthsCount-1)[2:])):
                temp[j] = int(BinInBalance[j])
            InBalance.append(temp)
            print(temp)
        
        # Идём по распределениям
        for i in range(len(InBalance)):
            GoodBalance = True
            res = self.BalanceEquation(InBalance[i])
            if np.sum(res) == 0: # Если время для учавствующих путей разное
                continue
            
            InBalanceIndex = InBalance[i].index(1)
            for j in range(len(InBalance[i])):
                if InBalance[i][j] == 0: # Если не входит в распределение
                    if self.PathTime(j, res) < self.PathTime(InBalanceIndex, res): # Для всех неиспользуемых время в пути не ниже используемых
                        GoodBalance = False
                        break
            
            if(GoodBalance): 
                AllBalances.append(res)
                
        return AllBalances
            
                    
                        
                        
                    
            
            
            
            
            
        
        
    
        
        
            
        
