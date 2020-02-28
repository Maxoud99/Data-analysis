import pandas as pd
import numpy as np
from array import *
import matplotlib.pyplot as plt
from scipy import stats
from pylab import *
df = pd.read_csv('cell2celltrain.csv')
df.head()
df.replace([np.inf,-np.inf],np.nan)
df=df.dropna(axis='index',how='any')
df2=pd.read_csv('cell2celltrain.csv')
df2.replace([np.inf,-np.inf],np.nan)
df2=df2.dropna(axis='index',how='any')
df2.drop(df2[df2['Churn']=='No'].index,inplace=True)
PCH=len(df2)
PCHT=len(df2)/len(df)
#print(PCHT)
ct=1
PDF=[]
CONDPDF=[]
PDFpara1=[]
PDFpara2=[]
CPDFpara1=[]
CPDFpara2=[]
while ct<58:
    name2=df2.columns[ct]
    type2=isinstance(df2.at[1,name2],str)
    if type2!=True:
        m1=np.mean(df2[name2])
        s1=np.std(df2[name2])
        m1,s1=stats.norm.fit(df2[name2])
        xmax= max(df2[name2])
        xmin=min(df2[name2])
        x2=np.linspace(xmin,xmax,1000)
        p=stats.norm.pdf(x2,m1,s1)
        loc1,q1=stats.expon.fit(df[name2])
        ex1=stats.expon.pdf(x2,loc1,q1)
       # df2[name2].hist(density=True, bins=100)
        r,h=np.histogram(df[name2],density=True, bins=1000)
        h2=stats.norm.pdf(r[1:],m1,s1)
        h3=stats.expon.pdf(r[1:],loc1,q1)
        mse1=np.square(np.subtract(h2,r[1:])).mean()
        mse2=np.square(np.subtract(h3,r[1:])).mean()
        if mse1<mse2:
            CPDFpara1.append(m1)
            CPDFpara2.append(s1)
            CONDPDF.append("norm")
        else:
            CPDFpara1.append(loc1)
            CPDFpara2.append(q1)
            CONDPDF.append("expon")
        plt.show()

        #plt.plot(x2,p,'g',linewidth=2)
        #plt.show()
       
    ct=ct+1



i=2
meanlist=[]
varlist=[]
pdflist=[]
##########################   JOINT PROBABILITY  #############################

jointstr=[]
jointstr=df.groupby(['Occupation','OwnsMotorcycle'])#doing the intersection between to col
sizeOfjoint=jointstr.size()#finding the size of the intersection
sizeOfDf=len(df)#finding the size of the normal col
jointProbabilityOftwoStr=sizeOfjoint/sizeOfDf#cal the joint probability
#print('the joint probability between  '+str(jointProbabilityOftwoStr))

##########################   CONDITIONAL PROBABILITY  #############################

condlist=df.groupby('Occupation')['OwnsMotorcycle'].value_counts() / df.groupby('Occupation')['OwnsMotorcycle'].count()
#print('the conditional probability between  '+str(condlist))



#######  FILTERING THE COLs ACCORDING TO BE STRING OR NUM & DOING SUM OPERATIONS ON THEM ######

#plt.hist2d(df['MonthlyRevenue'],df['TotalRecurringCharge'] , bins=100) #Joint PDF

#h,xedg,yedg=np.histogram2d(df.iloc[:,3],df.iloc[:,4], bins=(10,10), density=True)
#x,y=np.meshgrid(xedg[1:],yedg[1:])
#plt.figure()
#plt.contourf(x,y,h)


########################################
indp=[]
kk=df.corr()
u=1
c=1

while u<58:
    c=1
    name=df.columns[u]
    type=isinstance(df.at[2,name],str)
    if type!=True :
       while c<58:
          name2=df.columns[c]
          type2=isinstance(df.at[2,name2],str)
          if type2!=True:
             if abs(kk[name][name2])<0.300001:
                  indp.append(1)
             else:
                  indp.append(0)
          c=c+1
    u=u+1
c=1
while c<58:
    name2=df.columns[c]
    type2=isinstance(df.at[2,name2],str)
    if type2!=True:
        m=np.mean(df[name2])
        s=np.std(df[name2])
        m,s=stats.norm.fit(df[name2])
        xmax= max(df[name2])
        xmin=min(df[name2])
        x=np.linspace(xmin,xmax,1000)
        p=stats.norm.pdf(x,m,s)
        loc,scale=stats.expon.fit(df[name2])
        ex=stats.expon.pdf(x,loc,scale)
        #b1,b2,b3,b4=stats.beta.fit(df[name2])
        #beta=stats.beta.pdf(x,b1,b2,b3,b4)
        #df[name2].hist(density=True, bins=20)
        r,h=np.histogram(df[name2],density=True, bins=1000)
        h2=stats.norm.pdf(r[1:],m,s)
        h3=stats.expon.pdf(r[1:],loc,scale)
        #plt.plot(x,p,'k',linewidth=2)
        #plt.plot(x,ex,'r',linewidth=2)
        #plt.plot(x,beta,'y',linewidth=2)
        PDF.append(r)
        mse1=np.square(np.subtract(h2,r[1:])).mean()
        mse2=np.square(np.subtract(h3,r[1:])).mean()
        if mse1<mse2:
            PDFpara1.append(m)
            PDFpara2.append(s)
            PDF.append("norm")
        else:
            PDFpara1.append(loc)
            PDFpara2.append(scale)
            PDF.append("expon")
      #  plt.show()
        #print(name2)
        #print('normal')
        #print(mse1)
        #print('expon')
        #print(mse2)
    c=c+1
#print(indp)
#while i<58:
 #   flag=True
  #  j=df.columns[i]+''
   # type=isinstance(df.at[5,j],str)
    #arr=df[j]
    
 #   prob1=df.groupby(j)[j].transform(lambda x : x.count()/len(df))
  #  if type ==True :
   #     k=j+' probability'
    #    df[k] = prob1    
    #else :
     #   #df.hist(column=j) #calculating the histo of each col
      #  num_bins=20
       # counts, bin_edges = np.histogram(df[j], bins=num_bins, normed=True)
        #plt.show()
       
        #pdf
        #plt.figure()
        #df[j].hist(cumulative=1, density=True, bins=100)
        #plt.show()
        #cdf
        #plt.figure()
        #df[j].hist(cumulative=True, density=1, bins=100)
        #plt.show()
        #m=df[j].mean() #calculating the mean of each  col
        #n=str(m) #converting float to string
        #h='The mean of '+j+' is = '+n
        #meanlist.append(h)#appending the mean values in the list
        #s=df[j].var() #calculating the var of each col
        #b=str(s) #converting float to string
        #l='The variance of '+j+' is = '+b
        #varlist.append(l)#appending the var values in the list
    #i+=1
########################  PRINTING SOME OUTS  #####################################

#print(df)#printing the grid 
#print(meanlist)#printing the mean values
#print(varlist)#printing the variance values
######################  Anomalies  #######################################

    #print(indp)
i=1
Pnorm=0.0
Cnorm=0.0
bae=[]
while i<=1000:
    j=1
    nresult=1
    dresult=1
    
    while j<=3:
        name2=df.columns[j]
        type2=isinstance(df.at[2,name2],str)
        if type2!=True:
         p1=PDFpara1[j]
         p2=PDFpara2[j]
         cp1=CPDFpara1[j]
         cp2=CPDFpara2[j]
         l=PDF[j]
         k=CONDPDF[j]
         cell=df.iloc[i,j]
         if l=="norm":
            Pnorm=stats.norm.pdf(cell,p1,p2)
         else:
            Pnorm=stats.expon.pdf(cell,p1,p2)
         if k=="norm":
            Cnorm=stats.norm.pdf(cell,cp1,cp2)
         else:
            Cnorm=stats.expon.pdf(cell,cp1,cp2)
         
         nresult=nresult*Cnorm
         dresult=dresult*Pnorm
         
        j=j+1
    k=(nresult*PCHT)/dresult
    bae.append(k)
    i=i+1
print(bae)
#print(len(df))
#df.drop(df[df['AgeHH1']==0].index,inplace=True)
#df.drop(df[df['AgeHH2']==0].index,inplace=True)
#print(CONDPDF)
#print(len(CONDPDF))