#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import expon
import scipy.stats as stats




class Dataset:
    def __init__(self, path_to_csv):
        self.data = pd.read_csv(path_to_csv)


    def histogramas(self):
        for col in self.data:
            histo = sns.distplot(self.data[col])
            fig = histo.get_figure()
            fig.savefig('histogramas/Histogram_' + col + '.png')
            # fig.cla()
            fig.clf()

    def boxPlot(self):
        for col in self.data:
            data = pd.concat([self.data['VO2'], self.data[col]], axis=1)
            f, ax = plt.subplots(figsize=(8, 6))
            fig = sns.boxplot(y=col, data=self.data, width=0.2)
            fig.axis(ymin=2)
            plt.savefig('boxplots/BoxPlot_' + col + '.png')
            plt.cla()
            plt.clf()

    def paramet(self):
        for col in self.data:
            print col

            self.data[col].hist(density=True)
            l = self.mleExp(col)
            p = []
            fig, ax = plt.subplots()
            for i in range(int(math.floor(self.data[col].min())), int(math.ceil(self.data[col].max()))):
                p.append(self.expPmf(i, l))
            ax.plot(range(int(math.floor(self.data[col].min())), int(math.ceil(self.data[col].max()))), p, label="Exponencial")


            self.data[col].hist(density=True)
            gm, go2 = self.mleGauss(col)
            j = []
            for i in range(int(math.floor(self.data[col].min())), int(math.ceil(self.data[col].max()))):
                j.append(self.gaussPmf(i, gm, go2))
            ax.plot(range(int(math.floor(self.data[col].min())), int(math.ceil(self.data[col].max()))), j, label="Normal")
            # plt.savefig("histogram_pmf/"+col+"norm.png")
            # plt.cla()


            self.data[col].hist(density=True)
            m, o2 = self.mleLogNormal(col)
            j = []
            for i in range(int(math.floor(self.data[col].min())), int(math.ceil(self.data[col].max()))):
                j.append(self.lognormPmf(i, m, o2))
            ax.plot(range(int(math.floor(self.data[col].min())), int(math.ceil(self.data[col].max()))), j, label="Lognormal")

            legend = ax.legend()
            legend.get_frame()
            plt.savefig("histogram_pmf/" + col + ".png")
            plt.cla()

            plt.title('ProbabilityPlot - '+col+'_Expon')
            stats.probplot(self.data[col], dist="expon", sparams=(0,1.0/l), plot=plt)
            plt.savefig("PPPlot/"+col+"_expon.png")
            plt.cla()

            plt.title('ProbabilityPlot - '+col+'_Norm')
            stats.probplot(self.data[col], dist="norm", sparams=(gm, go2), plot=plt)
            plt.savefig("PPPlot/"+col+"_norm.png")
            plt.cla()

            plt.title('ProbabilityPlot - '+col+'_Lognorm')
            stats.probplot(self.data[col], dist="lognorm", sparams=(o2, math.exp(m)), plot=plt)
            plt.savefig("PPPlot/"+col+"_lognorm.png")
            plt.cla()


    def mleExp(self, col):

        print "Exponential"
        c = self.data[col].count()*1.0
        l = c / self.data[col].sum()
        print "lambda = ",l
        return l

    def mleGauss(self, col):
        m = 1.0*self.data[col].sum()/self.data[col].count()
        temp = 0
        for i in self.data[col]:
            temp += (i-m)**2
        o2 = temp*1.0/self.data[col].count()

        print "Gaussiam"
        print "média: ", m
        print "desvio padrao",o2
        return m, o2


    def mleLogNormal(self, col):
        temp1 = 0
        for i in self.data[col]:
            temp1 += math.log(i)
        m = temp1*1.0/self.data[col].count()

        temp2 = 0
        for i in self.data[col]:
            temp2 += (math.log(i) - m)**2

        o2 = (temp2/self.data[col].count())

        print "Lognormal"
        print "média = ", m
        print "desvio = ", o2
        return m, o2


    def expPmf(self, x, l):
        return l*math.exp(-l*x)

    def weib(self, x,lamb,k):
        return (k / lamb) * (x / lamb)**(k-1) * np.exp(-(x/lamb)**k)

    def gaussPmf(self, x, m, o2):
        return ((2*math.pi*o2)**(-0.5))*math.exp(-0.5*(((x - m)**2)/o2))

    def lognormPmf(self, x, m, o2):
        p1 = 1.0/(x*o2*math.sqrt(2*math.pi))
        p2 = (-(math.log(x)-m)**2)/(2*o2)
        return p1*math.exp(p2)

    def corr(self):
        r = []
        for col in self.data:
            r1 = []
            m = self.data[col].sum()*1.0/self.data[col].count()
            for col2 in self.data:
                p1 = 0
                p21 = 0
                p22 = 0
                m2 = self.data[col2].sum()*1.0/self.data[col2].count()

                for i in range(self.data[col].count()):
                    p1 += (float(self.data.iloc[i][col])-m)*(float(self.data.iloc[i][col2])-m2)
                    p21 += ((float(self.data.iloc[i][col])-m)**2)
                    p22 += ((float(self.data.iloc[i][col2])-m2)**2)

                r1.append((p1)/(math.sqrt(p21)*math.sqrt(p22)))
            r.append(r1)

        for i in r:
            print i
