import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pylab import *
from sympy import Symbol, Derivative
df = pd.read_csv('cell2celltrain.csv')
df.replace([np.inf,-np.inf],np.nan)
df=df.dropna(axis='index',how='any')

i=1
meanlist=[]
varlist=[]
pdflist=[]
##########################   JOINT PROBABILITY  #############################

jointstr=[]
jointstr=df.groupby(['Occupation','OwnsMotorcycle'])#doing the intersection between to col
sizeOfjoint=jointstr.size()#finding the size of the intersection
sizeOfDf=len(df)#finding the size of the normal col
jointProbabilityOftwoStr=sizeOfjoint/sizeOfDf#cal the joint probability
print('the joint probability between  '+str(jointProbabilityOftwoStr))

##########################   CONDITIONAL PROBABILITY  #############################

condlist=df.groupby('Occupation')['OwnsMotorcycle'].value_counts() / df.groupby('Occupation')['OwnsMotorcycle'].count()
print('the conditional probability between  '+str(condlist))



#######  FILTERING THE COLs ACCORDING TO BE STRING OR NUM & DOING SUM OPERATIONS ON THEM ######

while i<58:
    j=df.columns[i]+''
    type=isinstance(df.at[5,j],str)
    if type ==True :
        k=j+' probability'
        df[k] = df.groupby(j)[j].transform(lambda x : x.count()/len(df))    
    else :
        #cumulative = np.cumsum(sorted_data)
        #df.hist(column=j) #calculating the histo of each col
        num_bins=20
        counts, bin_edges = np.histogram(df[j], bins=num_bins, normed=True)
        #plt.show()
        #cdf = np.cumsum(counts)
        
        #pdf
        #s = pd.Series(df[j].values)
        #plt.figure()
        #df[j].plot.kde()
        #plt.show()
        #cdf
        #plt.figure()
        #df[j].hist(cumulative=True, density=1, bins=100)
        #plt.show()
        
        m=df[j].mean() #calculating the mean of each  col
        n=str(m) #converting float to string
        h='The mean of '+j+' is = '+n
        meanlist.append(h)#appending the mean values in the list
        s=df[j].var() #calculating the var of each col
        b=str(s) #converting float to string
        l='The variance of '+j+' is = '+b
        varlist.append(l)#appending the var values in the list
    i+=1
########################  PRINTING SOME OUTS  #####################################
df['OutboundCalls'].plot(df['OutboundCalls'], norm.pdf(df['OutboundCalls']),'r-', lw=5, alpha=0.6, label='norm pdf')
df['OutboundCalls'].plot.kde()
print(df)#printing the grid 
print(meanlist)#printing the mean values
print(varlist)#printing the variance values


