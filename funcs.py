import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tools.plotting import *
import seaborn
np.set_printoptions(3,suppress=1)

#=====================================================================
#=================DATA-INPUT==========================================
#=====================================================================
def ReadData(datafile):
    """
    Return ID,real,res from datafile	
    """
    r =pd.read_table(datafile,skiprows=8,chunksize=3)
    i=0
    ID=[]
    real=[]
    res=[]
    for chunk in r:
        ID.append(chunk.ix[0][0][1:])
        real.append(np.fromstring(chunk.ix[1][0],sep=' '))
        res.append(np.fromstring(chunk.ix[2][0],sep=' '))
        i=i+1
    real=np.array(real)
    res=np.array(res)
    return ID,real,res


#=====================================================================
#=================INTERNAL-FUNCTIONS==========================================
#=====================================================================
def MMean(real,res,i1,i2):
    """
    Internal Function	
    """
    S=0.0
    N=0
    for i in range(len(real)):
        if real[i][i1]==1:
            S=S+res[i][i2]
            N=N+1
    return S/N

def MSD(real,res,i1,i2):
    """
    Internal Function	
    """
    S=0.0
    N=0
    m = MMean(real,res,i1,i2)
    for i in range(len(real)):
        if real[i][i1]==1:
            S=S+(res[i][i2]-m)**2
            N=N+1
    return np.sqrt(S/N)

def Hist(real,res,i1,i2):
    S=[]
    for i in range(len(real)):
        if real[i][i1]==1:
            S.append(res[i][i2])
    S = np.array(S)
    w= np.ones_like(S)/len(S)
    return S,w
#=====================================================================
#=================MATRIX-METRICS==========================================
#=====================================================================
def WTA(ID,real,res):
    """
    Return Winner Takes All Matrix from ID,real,res	
    """
    CM=np.zeros((4,4),dtype=int)
    for i in range(len(ID)):
        CM[real[i].argmax(),res[i].argmax()]=CM[real[i].argmax(),res[i].argmax()]+1
    return CM

def ConfusionMatrix(ID,real,res,TP=0.5,FP=0.5):
    """
    Return [0]Numpy Array CM,[1]Numpy Array NormCM,[2]Pandas Dataframe CM,[3]Pandas Dataframe NormCM  from ID,real,res	
    """
    Ntypes = np.zeros((4,1))
    CM=np.zeros((4,4),dtype=int)
    for i in range(len(ID)):
        p=real[i].argmax()
        Ntypes[p]=Ntypes[p]+1
            
        if (res[i][p]>=TP):
            CM[p,p]=CM[p,p]+1
            
        for n in range(4):
            if ((n!=p) and (res[i][n]>=FP)):
                    CM[p,n]=CM[p,n]+1
    NCM=CM/Ntypes
    dfCM=pd.DataFrame.from_records(CM,index=['E','S','I','Q'],columns=['E(unit)','S(unit)','I(unit)','Q(unit)'])
    dfNCM=pd.DataFrame.from_records(NCM,index=['E','S','I','Q'],columns=['E(unit)','S(unit)','I(unit)','Q(unit)'])
    return CM,NCM,dfCM,dfNCM


def AverageMatrix(real,res):
    """
    Return [0]np.Array AverageMatrix,[1]npArray SDMatrix,[2]Pandas Dataframe AverageMatrix,[3]Pandas Dataframe SDMatrix  from ID,real,res	
    """
    MM=np.zeros((4,4))
    DM=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            MM[i,j]=MMean(real,res,i,j)
            DM[i,j]=MSD(real,res,i,j)
    dfMM=pd.DataFrame.from_records(MM,index=['E','S','I','Q'],columns=['E(unit)','S(unit)','I(unit)','Q(unit)'])
    dfDM=pd.DataFrame.from_records(DM,index=['E','S','I','Q'],columns=['E(unit)','S(unit)','I(unit)','Q(unit)'])
    return MM,DM,dfMM,dfDM


#=====================================================================
#=================PLOT-FUNCTIONS==========================================
#=====================================================================
    #fig = plt.figure(figsize=(5,5))
    #ax = fig.add_subplot(111)
    #ax.grid(False)
    #ax.set_xticklabels(['E', 'S', 'I', 'Q'] )
    #ax.set_yticklabels(['E', 'S', 'I', 'Q'] )
    #ax.matshow(df,cmap=plt.cm.GnBu)
    #for (i, j), z in np.ndenumerate(df):
    #    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    #plt.tight_layout()

def plot_CM(df,datafile,TP,FP):
    ax=seaborn.hetamap(df,annot=True)
    ax.set_title("Confusion matrix (TP> %0.1f, DP> %0.1f) for file: %s"%(TP,FP,datafile),fontsize=12)
    

def plot_MM(dfM,dfsd,datafile):
    ax=seaborn.heathmap(dfM,annot=True)
    ax.set_title("Average for file: %s"%(datafile),fontsize=12)
    #for (i, j), z in np.ndenumerate(dfM):
    #    ax.text(j, i, '{:0.2f} +/- {:0.2f}'.format(z,dfsd.ix[i,j]), ha='center', va='center')
    #plt.tight_layout()

def plot_CM_MM(dfcm,dfM,dfsd,datafile,TP,FP):
    fig = plt.figure(figsize=(16,10))
    ax1 = plt.subplot2grid((1,3), (0,0))
    ax1.grid(False)
    ax1.set_xticklabels(['','E', 'S', 'I', 'Q'] )
    ax1.set_yticklabels(['','E', 'S', 'I', 'Q'] )
    ax1.matshow(dfcm,cmap=plt.cm.GnBu)
    ax1.set_title("Confusion matrix (TP> %0.1f, DP> %0.1f) for file: %s"%(TP,FP,datafile),fontsize=12)
    for (i, j), z in np.ndenumerate(dfcm):
        ax1.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    
    ax2 = plt.subplot2grid((1,3), (0,2))
    ax2.grid(False)
    ax2.set_xticklabels(['','E', 'S', 'I', 'Q'] )
    ax2.set_yticklabels(['','E', 'S', 'I', 'Q'] )
    ax2.matshow(dfM,cmap=plt.cm.GnBu)
    ax2.set_title("Average Matrix for file: %s"%(datafile),fontsize=12)
    for (i, j), z in np.ndenumerate(dfM):
        ax2.text(j, i, '{:0.2f} +/- {:0.2f}'.format(z,dfsd.ix[i,j]), ha='center', va='center')
    plt.tight_layout()
    
def plot_bars(df):  
    fig=plt.figure(figsize=(18,5))
    ax1 = plt.subplot2grid((1,2), (0,0))
    ax2 = plt.subplot2grid((1,2), (0,1))
    #ax.set_title("BarPlot for file: %s"%datafile,fontsize=12)
    df.plot(kind='bar',ax=ax1)
    df.plot(kind='bar',stacked='true',ax=ax2)
    plt.tight_layout()

def make_hists(real,res):
    X=4
    Y=4
    fig =plt.figure(figsize=(16,8))
    fig.text(-0.035,0.125,'Histograms For \n Galaxy Type Q', ha="center", va="center",size=12,bbox=dict(boxstyle="square", alpha=0.5))
    fig.text(-0.035,0.375,'Histograms For \n Galaxy Type I', ha="center", va="center",size=12,bbox=dict(boxstyle="square", alpha=0.5))
    fig.text(-0.035,0.625,'Histograms For \n Galaxy Type S', ha="center", va="center",size=12,bbox=dict(boxstyle="square", alpha=0.5))
    fig.text(-0.035,0.875,'Histograms For \n Galaxy Type E', ha="center", va="center",size=12,bbox=dict(boxstyle="square", alpha=0.5))

    fig.text(0.125,1.025,'E - Unit',rotation=270, ha="center",va="center",size=12,bbox=dict(boxstyle="rarrow", alpha=0.5))
    fig.text(0.375,1.025,'S - Unit',rotation=270, ha="center", va="center",size=12,bbox=dict(boxstyle="rarrow,pad=0.3", alpha=0.5))
    fig.text(0.625,1.025,'I - Unit',rotation=270, ha="center", va="center",size=12,bbox=dict(boxstyle="rarrow,pad=0.3", alpha=0.5))
    fig.text(0.875,1.025,'Q - Unit',rotation=270, ha="center", va="center",size=12,bbox=dict(boxstyle="rarrow,pad=0.3", alpha=0.5))

    ax1 = plt.subplot2grid((X,Y), (0,0))
    ax2 = plt.subplot2grid((X,Y), (0,1))
    ax3 = plt.subplot2grid((X,Y), (0,2))
    ax4 = plt.subplot2grid((X,Y), (0,3))

    ax5 = plt.subplot2grid((X,Y), (1,0))
    ax6 = plt.subplot2grid((X,Y), (1,1))
    ax7 = plt.subplot2grid((X,Y), (1,2))
    ax8 = plt.subplot2grid((X,Y), (1,3))

    ax9 = plt.subplot2grid((X,Y), (2,0))
    ax10 = plt.subplot2grid((X,Y), (2,1))
    ax11 = plt.subplot2grid((X,Y), (2,2))
    ax12 = plt.subplot2grid((X,Y), (2,3))

    ax13 = plt.subplot2grid((X,Y), (3,0))
    ax14 = plt.subplot2grid((X,Y), (3,1))
    ax15 = plt.subplot2grid((X,Y), (3,2))
    ax16 = plt.subplot2grid((X,Y), (3,3))

    h,w=Hist(real,res,0,0)
    ax1.hist(h,weights=w)
    ax1.set_ylabel('E (TP)')

    h,w=Hist(real,res,0,1)
    ax2.hist(h,weights=w)
    ax2.set_ylabel('S')

    h,w=Hist(real,res,0,2)
    ax3.hist(h,weights=w)
    ax3.set_ylabel('I')

    h,w=Hist(real,res,0,3)
    ax4.hist(h,weights=w)
    ax4.set_ylabel('Q')

    h,w=Hist(real,res,1,0)
    ax5.hist(h,weights=w)
    ax5.set_ylabel('E')

    h,w=Hist(real,res,1,1)
    ax6.hist(h,weights=w)
    ax6.set_ylabel('S (TP)')

    h,w=Hist(real,res,1,2)
    ax7.hist(h,weights=w)
    ax7.set_ylabel('I')

    h,w=Hist(real,res,1,3)
    ax8.hist(h,weights=w)
    ax8.set_ylabel('Q')

    h,w=Hist(real,res,2,0)
    ax9.hist(h,weights=w)
    ax9.set_ylabel('E')

    h,w=Hist(real,res,2,1)
    ax10.hist(h,weights=w)
    ax10.set_ylabel('S')

    h,w=Hist(real,res,2,2)
    ax11.hist(h,weights=w)
    ax11.set_ylabel('I (TP)')

    h,w=Hist(real,res,2,3)
    ax12.hist(h,weights=w)
    ax12.set_ylabel('Q')

    h,w=Hist(real,res,3,0)
    ax13.hist(h,weights=w)
    ax13.set_ylabel('E')

    h,w=Hist(real,res,3,1)
    ax14.hist(h,weights=w)
    ax14.set_ylabel('S')

    h,w=Hist(real,res,3,2)
    ax15.hist(h,weights=w)
    ax15.set_ylabel('I')

    h,w=Hist(real,res,3,3)
    ax16.hist(h,weights=w)
    ax16.set_ylabel('Q (TP)')
    plt.tight_layout()
#plt.savefig('hist.pdf')

#=====================================================================
#=================MAIN-REPORTS==========================================
#=====================================================================
def FullReport(datafile,TP=0.5,FP=0.5):
    ID,real,res=ReadData(datafile)
    wtaM=WTA(ID,real,res)
    CM,NCM,dfM,dfN = ConfusionMatrix(ID,real,res,TP,FP)
    MM,DM,dfMM,dfDM = AverageMatrix(real,res)
    plot_CM_MM(dfN,dfMM,dfDM,datafile,TP,FP)
    plot_bars(dfN)
    make_hists(real,res)
