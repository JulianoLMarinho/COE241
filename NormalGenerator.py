import numpy as np
from math import pi
import matplotlib.pyplot as plt

def vaNormal(n):
	Tntry = 0
	x = []
	xy = []
	xNP = []
	for i in range(n):
		accept = True
		ntry = 0;
		xNP.append(np.random.normal(0,1))
		while accept:
			u1 = np.random.uniform(0,1)
			u2 = np.random.uniform(0,1)
			y = np.random.exponential(1)
			g = np.exp(-((y-1.0)**2.0)/2.0)
			ntry+=1
			if u1 <= g:
				if u2 > 0.5:
					x.append(y)
					xy.append(g)
				else:
					x.append(-y)
					xy.append(g)
				accept = False
		Tntry+=ntry
	plt.plot(x, xy, linestyle='--', color='b', marker='s', 
         linewidth=0.0)
	plt.show()
	#print "x = ",x,"\nMTentativas = ",Tntry*1.0/n,"\nxNP = ",xNP
	return x, xNP

c = (2.0*np.exp(1.0)/pi)**(1.0/2.0)


vaNormal(100000)
