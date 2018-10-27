#!/usr/bin/env python
# -*- coding: utf-8 -*-
from asn1crypto.core import Concat
import time
import matplotlib.pyplot as plt


def CongruenteLinear(Xo = 4, r = 10, a = 25214903917, M = 2**48, c = 11):
    """Obtém uma lista de números pseudo-aleatórios onde Xo é a semente e r é o tamanho da lista de números retornada"""
    U = [Xo*1.0/M]
    X = [Xo*1.0]
    n = 1
    while(n<r):
        xi = (a*X[-1]+c)%M
        U.append(xi/M)
        X.append(xi)
        n+=1
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
    print "Média teórica = ", Eteorico, "\nMédia experimental = ", Esimulado

    return 1




pmfT = [0.11, 0.09, 0.11, 0.09, 0.11, 0.09, 0.11, 0.09, 0.11, 0.09]
pmfX = [5,6,7,8,9,10,11,12,13,14]
print SimulateXPMF(4,5000,pmfT, pmfX)

