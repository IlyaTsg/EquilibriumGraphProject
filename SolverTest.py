# Правильный ввод зависимости: 10x + 12 или 0x + 12 или 3x

EDGES_COUNT = 5
PATHS_COUNT = 3
FLOW = 6 # Общий поток

import numpy as np
from scipy.linalg import solve

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
def EqualTime(EdgesExpressionList, Paths, NpathA, NpathB):
    coefs = np.array([0]*PATHS_COUNT)
    frees = np.array([0])
    
    for i in range(EDGES_COUNT):
        if Paths[i][NpathA]: # Если ребро учавствует в пути
            for j in range(PATHS_COUNT): # Оставльные потоки тоже нужно учитывать
                if Paths[i][j]:
                    coefs[j] += EdgesExpressionList[i].coef
                    frees[0] -= EdgesExpressionList[i].free
        if Paths[i][NpathB]:
            for j in range(PATHS_COUNT): # Минусуем все переменные, т.к перенос
                if Paths[i][j]:
                    coefs[j] -= EdgesExpressionList[i].coef
                    frees[0] += EdgesExpressionList[i].free
    
    print("DEBUG func(EqualTime)", coefs)
    print("DEBUG func(EqualTime)", frees)
    
    result = []
    result.append(coefs)
    result.append(frees)
    return result
                
EdgesExpressionList = []
for i in range(EDGES_COUNT):
    EdgesExpressionList.append(Edge(input("Enter expression: ")))

Paths = [[1, 0, 0],
         [0, 1, 1],
         [1, 0, 1],
         [0, 1, 0],
         [0, 0, 1]]

result = EqualTime(EdgesExpressionList, Paths, 0, 1)
coefs = result[0]
frees = result[1]

result = EqualTime(EdgesExpressionList, Paths, 0, 2)
coefs1 = result[0]
frees1 = result[1]

coefs = np.append(coefs, coefs1)
frees = np.append(frees, frees1)

coefs = np.append(coefs, [1, 1, 1])
frees = np.append(frees, FLOW)

coefs = np.reshape(coefs, (3, 3))
frees = np.reshape(frees, (3, 1))

print(coefs)
print(frees)

res = solve(coefs, frees)

print(res)