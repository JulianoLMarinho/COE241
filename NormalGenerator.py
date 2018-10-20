#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats


def vaNormal(n):
	Tntry = 0
	x = []
	xy = []
	xNP = []
	xyNP = []
	for i in range(n):
		accept = True
		ntry = 0;
		tempNP = np.random.normal(0,1)
		xNP.append(tempNP)
		xyNP.append(2.0*np.exp(-((tempNP*1.0)**2.0)/2.0)/((2*pi)**(0.5)))
		while accept:
			u1 = np.random.uniform(0,1)
			u2 = np.random.uniform(0,1)
			y = np.random.exponential(1)
			g = np.exp(-(((y*1.0)-1.0)**2.0)/2.0)
			gN = 2.0*np.exp(-((y*1.0)**2.0)/2.0)/((2*pi)**(0.5))
			ntry+=1
			if u1 <= g:
				if u2 > 0.5:
					x.append(y)
					xy.append(gN)
				else:
					x.append(-y)
					xy.append(gN)
				accept = False
		Tntry+=ntry
	plt.plot(x, xy, linestyle='--', color='b', marker='s', markersize = 0.5, 
         linewidth=0.0)
	plt.title('Grafico com as amostras do gerador')
	plt.xlabel('x')
	plt.ylabel('f(x)')
	plt.savefig("AmostrasGerador.png")
	plt.cla()
	plt.title('Grafico com as amostras do numpy')
	plt.xlabel('x')
	plt.ylabel('f(x)')
	plt.plot(xNP, xyNP, linestyle='--', color='b', marker='s', markersize = 0.5, 
         linewidth=0.0)
	plt.savefig("AmostrasNP.png")
	plt.cla()
	stats.probplot(x, dist="norm", plot=plt)
	plt.savefig('QQplot.png')
	print "c calculado = ", Tntry*1.0/n, "\nc teórico = ",(2.0*np.exp(1.0)/pi)**(1.0/2.0)
	return x, xNP

c = (2.0*np.exp(1.0)/pi)**(1.0/2.0)


vaNormal(10000)
