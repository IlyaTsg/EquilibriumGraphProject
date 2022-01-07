""" Класс графа """

from os import PathLike, execlp
from typing import NewType
import networkx as nx
from networkx.algorithms.shortest_paths import weighted
from networkx.generators.random_graphs import newman_watts_strogatz_graph
from networkx.readwrite import edgelist
from numpy.lib.function_base import append
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy.linalg import solve

MAX = 9999.1 # Условный максимум

class Graph:
    StartVertices = [] # Стратовые вершины
    FinishVertices = [] # Финишные вершины
    Matrix = [] # Матрица смежности
    Puths = [] # Список всех путей от истока до стока
    PuthsMatrix = [] # Матрица путей для рёбер
    EdgesExpressionList = [] # Список функций для рёбер
    Flow = 0
    BalanceTime = 0 # Общее время в пути для равновесия
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
                    temp_F = int(temp[0:len(temp)-1])-1
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
    
    def ResetGraph(self):
        self.Puths = []
        self.PuthsMatrix = []
        self.EdgesExpressionList = []
        self.Flow = 0
        self.BalanceTime = 0
        self.PuthsCount = 0
    
    def SetFlow(self, flow):
        self.Flow = flow
            
    def ShowGraph(self):
        copy_start = []
        copy_finish = []
        
        for i in range(self.EdgesCount): # На рисунке номера вершин больше на 1
            copy_start.append(self.StartVertices[i]+1)
            copy_finish.append(self.FinishVertices[i]+1)
        
        DF = pd.DataFrame({'S': copy_start, 'F': copy_finish}) # Формируем dataframe рёбер
        
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
        self.PuthsMatrix = []
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
                    self.free = np.append(self.free, float(expression.split()[2])*(-1))
                else:
                    self.free = np.append(self.free, float(expression.split()[2]))
            else:
                self.free = np.append(self.free, float(0))
    
    def PrepareExpressionList(self, text: str):
        j = 0
        for i in range(self.EdgesCount):
            exp = ""
            while  j != len(text) and text[j] != "\n":    
                exp += text[j]
                j += 1
            j += 1
            self.EdgesExpressionList.append(self.Edge(exp))
            
    # Составление и решение уравнения для двух путей
    def EqualTime(self, EdgesExpressionList, Paths, InBalance: list, NpathA, NpathB):
        coefs = np.array([0]*self.PuthsCount)
        frees = np.array([0], dtype="float")
        
        for i in range(self.EdgesCount):
            if Paths[i][NpathA]: # Если ребро учавствует в пути
                for j in range(self.PuthsCount): # Оставльные потоки тоже нужно учитывать
                    if Paths[i][j] and InBalance[j]:
                        coefs[j] += EdgesExpressionList[i].coef
                frees[0] -= EdgesExpressionList[i].free
            if Paths[i][NpathB]:
                for j in range(self.PuthsCount): # Минусуем все переменные, т.к перенос
                    if Paths[i][j] and InBalance[j]:
                        coefs[j] -= EdgesExpressionList[i].coef
                frees[0] += EdgesExpressionList[i].free
        
        result = []
        result.append(coefs)
        result.append(frees)
        return result
            
    def BalanceEquation(self, InBalance: list): # Поиск равновесного распределения
        coefs = np.array([])
        frees = np.array([], dtype="float")
        EquationCount = 0
        
        for i in range(self.PuthsCount):
            if InBalance[i] == 1:
                for j in range(i+1, self.PuthsCount):
                    if InBalance[j] == 1:
                        result = self.EqualTime(self.EdgesExpressionList, self.PuthsMatrix, InBalance, i, j)
                        coefs = np.append(coefs, result[0])
                        frees = np.append(frees, result[1])
                        EquationCount += 1
                break
                
        coefs = np.append(coefs, InBalance)
        frees = np.append(frees, float(self.Flow))
        EquationCount += 1
        
        for j in range(len(InBalance)):
            temp = [0]*self.PuthsCount
            if InBalance[j] == 0:
                temp[j] = 1
                coefs = np.append(coefs, np.array(temp))
                frees = np.append(frees, float(0))
        
        coefs = np.reshape(coefs, (self.PuthsCount, self.PuthsCount))
        frees = np.reshape(frees, (self.PuthsCount, 1))
        
        try:
            res = solve(coefs, frees)
        except np.linalg.LinAlgError:
            res = [-1]*len(InBalance)
        
        return res
    
    def PathTime(self, PuthNum, Balance: list, mode):
        res = 0
        if mode == 1:
            for i in range(self.EdgesCount):
                if self.PuthsMatrix[i][PuthNum] == 1:
                    for j in range(self.PuthsCount):
                        if self.PuthsMatrix[i][j] == 1:
                            if Balance[j]:
                                res += self.EdgesExpressionList[i].coef[0]*Balance[j]
                                res += self.EdgesExpressionList[i].free
        else:
            for i in range(self.EdgesCount):
                if self.PuthsMatrix[i][PuthNum] == 1:
                    for j in range(self.PuthsCount):
                        if self.PuthsMatrix[i][j] == 1:
                            res += self.EdgesExpressionList[i].coef[0]*Balance[j]
                            if(j == PuthNum or Balance[j] != 0):
                                res += self.EdgesExpressionList[i].free
        return res
    
    # Поиск равновесия
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
        
        # Идём по распределениям
        for i in range(len(InBalance)):
            GoodBalance = True
            res = self.BalanceEquation(InBalance[i])
            
            for j in res: # Все потоки не отрицательны
                if j < 0:
                    GoodBalance = False
                    break
            
            for road in range(len(res)):
                if res[road] > 0:
                    InBalanceIndex = road
                    break
                    
            for j in range(self.PuthsCount):
                if res[j] == 0: # Если не входит в распределение
                    if self.PathTime(j, res, 0) < self.PathTime(InBalanceIndex, res, 0): # Для всех неиспользуемых время в пути не ниже используемых
                        GoodBalance = False
                        break
            
            if self.PathTime(InBalanceIndex, res, 1) < 0:
                GoodBalance = False        
                
            if GoodBalance: 
                AllBalances.append(res)
                self.BalanceTime = self.PathTime(InBalanceIndex, res, 1)
                    
        return AllBalances
        
    # Поиск неэффективной дороги
    def FindUneffective(self):
        UneffectiveEdges = []
        OLD_TIME = self.BalanceTime
        for i in range(self.EdgesCount):
            temp = self.EdgesExpressionList[i].free # Запоминаем свободный член
            self.EdgesExpressionList[i].free = np.array(MAX) # Закрываем дорогу
        
            NewAllBalances = self.FindBalance()
        
            if self.BalanceTime < OLD_TIME: # Если общее время меньше времени равновесия
                temp_time = []
                temp_time.append(i)
                temp_time.append(self.BalanceTime)
                UneffectiveEdges.append(temp_time)
            
            self.EdgesExpressionList[i].free = temp # Открываем дорогу, чтобы проверить другие
            self.FindBalance()
            
        if len(UneffectiveEdges) > 1:
            old_frees = []
            
            for i in range(len(UneffectiveEdges)):
                old_frees.append(self.EdgesExpressionList[UneffectiveEdges[i][0]].free)
                self.EdgesExpressionList[UneffectiveEdges[i][0]].free = MAX # Закрываем все дороги
            self.FindBalance()
            UneffectiveEdges.append(self.BalanceTime)
            
            for i in range(len(UneffectiveEdges)-1):
                self.EdgesExpressionList[UneffectiveEdges[i][0]].free = old_frees[i] # Открываем все дороги
            self.FindBalance()
        
        return UneffectiveEdges
        