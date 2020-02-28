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
#################################### probability of churn #############################
df2=pd.read_csv('cell2celltrain.csv')
df2.replace([np.inf,-np.inf],np.nan)
df2=df2.dropna(axis='index',how='any')
df2.drop(df2[df2['Churn']=='No'].index,inplace=True)
PCHT=len(df2)/49200
################################ correlation & covariance ##########################
correlation=df.corr()
covariance=df.cov()
################################# anomilies #######################################3

#################################### checking dependency #########################
u=1
indp=[]

while u<58:
    flag="indp"
    c=1
    name=df.columns[u]
    type=isinstance(df.at[2,name],str)
    if type!=True :
       while c<58:
          name2=df.columns[c]
          type2=isinstance(df.at[2,name2],str)
          if type2!=True:
             if abs(correlation[name][name2])<0.50001 and correlation[name][name2] !=1.0 :
                  flag="indp"
             else:
                  flag="dep"
                  break
          c=c+1
       indp.append(flag)
    u=u+1
   
###################################### fitting PDF ################################
PDF=[]
CONDPDF=[]
PDFpara1=[]
PDFpara2=[]
CPDFpara1=[]
CPDFpara2=[]
c=1
while c<58:
    name2=df.columns[c]
    type2=isinstance(df.at[2,name2],str)
    if type2!=True:
        m,s=stats.norm.fit(df[name2])
        xmax= max(df[name2])
        xmin=min(df[name2])
        x=np.linspace(xmin,xmax,10000)
        p=stats.norm.pdf(x,m,s)
        loc,scale=stats.expon.fit(df[name2])
        ex=stats.expon.pdf(x,loc,scale)
        h,r=np.histogram(df[name2],density=True, bins=10000)
        h2=stats.norm.pdf(r[1:],m,s)
        h3=stats.expon.pdf(r[1:],loc,scale)
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
    c+=1
###################################### fitting CPDF ################################
ct=1
while ct<58:
    name2=df2.columns[ct]
    type2=isinstance(df2.at[1,name2],str)
    if type2!=True:
        
        m1,s1=stats.norm.fit(df2[name2])
        xmax= max(df2[name2])
        xmin=min(df2[name2])
        x2=np.linspace(xmin,xmax,10000)
        p=stats.norm.pdf(x2,m1,s1)
        loc1,scale1=stats.expon.fit(df[name2])
        ex1=stats.expon.pdf(x2,loc1,scale1)
        h,r=np.histogram(df[name2],density=True, bins=10000)
        h2=stats.norm.pdf(r[1:],m1,s1)
        h3=stats.expon.pdf(r[1:],loc1,scale1)
        mse1=np.square(np.subtract(h2,r[1:])).mean()
        mse2=np.square(np.subtract(h3,r[1:])).mean()
        if mse1<mse2:
            CPDFpara1.append(m1)
            CPDFpara2.append(s1)
            CONDPDF.append("norm")
        else:
            CPDFpara1.append(loc1)
            CPDFpara2.append(scale1)
            CONDPDF.append("expon")
    ct=ct+1
################################ bayes rule #####################################
i=0
counter=0
bayes=[]
while i<49200:
    result=1
    j=1
    while j<=33:
        name2=df.columns[j]
        type2=isinstance(df.at[2,name2],str)
        check=indp[j]
        if check=="indp":
         if type2!=True:
          p1=PDFpara1[j]
          p2=PDFpara2[j]
          cp1=CPDFpara1[j]
          cp2=CPDFpara2[j]
          l=PDF[j]
          k=CONDPDF[j]
          cell=df.iloc[i,j]
          if l=="norm":
            denom=stats.norm.pdf(cell,p1,p2)
          else:
            denom=stats.expon.pdf(cell,p1,p2)
          if k=="norm":
            nom=stats.norm.pdf(cell,cp1,cp2)
          else:
            nom=stats.expon.pdf(cell,cp1,cp2)
          result=result*nom/denom
        j+=1
    k=result*PCHT
    if k>0.5:
        counter+=1
    bayes.append(k)
    i+=1
print(counter*100/49200)
print(PCHT*100)