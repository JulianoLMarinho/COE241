#!/usr/bin/env python
# -*- coding: utf-8 -*-
from asn1crypto.core import Concat
import time

def CongruenteLinear(Xo = 4, r = 10, a = 25214903917, M = 2**48, c = 11):
    """Obtém uma lista de números pseudo-aleatórios onde Xo é a semente e r é o tamanho da lista de números retornada"""
    U = [Xo*1.0/M]
    X = [Xo*1.0]
    n = 0
    while(n<r):
        xi = (a*X[-1]+c)%M
        U.append(xi/M)
        X.append(xi)
        n+=1
    return U

def SimulateXPMF(seed, r, pmf):
    """"""
