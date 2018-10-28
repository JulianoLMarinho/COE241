#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt


def CongruenteLinear(Xo = 4, r = 10, range = (0,1), a = 25214903917, M = 2**48, c = 11):
    """Obtém uma lista de números pseudo-aleatórios onde Xo é a semente e r é o tamanho da lista de números retornada"""
    U = [Xo*1.0/M]
    X = [Xo*1.0]
    n = 1
    while(n<r):
        xi = (a*X[-1]+c)%M
        U.append(xi/M)
        X.append(xi)
        n+=1
    U = np.interp(U, (0, 1), range)
    return U


def SimulateXPMF(seed, r, pmf, pmfX):
    """Simular os valores de X a partir da sua pmf"""
    somaPMF = []
    for i in range(len(pmf)):
        somaPMF.append(sum(pmf[0:i+1]))
    U = CongruenteLinear(seed, r)
    S = []
    for i in range(r):
        for j in pmfX:
            if U[i]<somaPMF[pmfX.index(j)]:
                S.append(j)
                break

    Eteorico = 0
    for i in range(len(pmf)):
        Eteorico += pmf[i]*pmfX[i]
    Esimulado = sum(S)*1.0/len(S)
    print "Média teórica = ", Eteorico, "\nMédia simulada = ", Esimulado

    return 1

#Função da questão 2
def func2(u):
    """Intervalo 1/(e-1)<=u<=e/(e-1)"""
    return math.log(u*(math.exp(1)-1))

#Funções para a questão 3
def func3a(u):
    """intervalo de 0 < u < e/2"""
    return math.log(2*u)/2

def func3b(u):
    """intervalo de -e/2 < u < 0"""
    return math.log(-2 * u) / -2

def func4(x, l):
    c = (2.0/l)*(math.exp(2.0*l-1))
    return ((x**2.0)*math.exp(-x*(1.0-l)))/(2.0*c*l)

def randomVariable(questao, seed, t, range):
    """Questões 2 e 3"""
    U = CongruenteLinear(seed, t, range)
    X = []
    for i in U:
        if questao == 2:
            X.append(func2(i))
        if questao == 3:
            if 0<=i and i<=math.exp(1)/2:
                X.append(func3a(i))
            if -math.exp(1)/2 <= i and i < 0:
                X.append(func3b(i))
    if questao==2:
        print "Média teórica = 0.58198", "\nMédia simulada = ", sum(X)/len(X)
    if questao==3:
        print "Média teórica = 0", "\nMédia simulada = ", sum(X)/len(X)



def acceptanceRejection(r, l):
    """Gera uma lista de va com distribuição func4, tamanho r e lamda l"""
    X = []
    while len(X)<r:
        y = np.random.exponential(l)
        u1 = np.random.uniform(0, 1)
        if u1<=func4(y, l):
            X.append(y)
    # plt.hist(X, 50)
    # plt.show()

    print "Média simulada = ", sum(X)*1.0/len(X)
    return 0


print "Questão 1:\n"
pmfT = [0.11,0.09,0.11,0.09,0.11,0.09,0.11,0.09,0.11,0.09]
pmfX = [5,6,7,8,9,10,11,12,13,14]
print SimulateXPMF(4,50000,pmfT, pmfX)
print "\nQuestão 2:\n"
randomVariable(2, 234, 50000, (1/(math.exp(1)-1), math.exp(1)/(math.exp(1)-1)))
print "\nQuestão 3:\n"
randomVariable(3, 7, 50000, (-math.exp(1)/2, math.exp(1)/2))
print "\nQuestão 4:\n"
acceptanceRejection(50000, 1)
# print CongruenteLinear(4, 20, (0, math.exp(1)))
