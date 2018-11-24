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

    def brincs(self):
        l = 1.0 / self.data["IDADE"].sum()
        c = self.data["IDADE"].count()
        p = []
        for i in self.data["IDADE"]:
            p.append(c/self.expPmf(i*1.0, l))
        return p





    def expPmf(self, x, l):
        return l*math.exp(-l*x)
