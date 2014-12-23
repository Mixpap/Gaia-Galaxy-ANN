import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tools.plotting import *
import seaborn
from os import listdir
np.set_printoptions(3, suppress=1)

#=====================================================================
#=================DATA-INPUT==========================================
#=====================================================================

def ReadData(datafile):
    """
    Return ID,real,res from datafile
    """
    r = pd.read_table(datafile, skiprows=8, chunksize=3)
    i = 0
    ID = []
    real = []
    res = []
    for chunk in r:
        ID.append(chunk.ix[0][0][1:])
        real.append(np.fromstring(chunk.ix[1][0], sep=' '))
        res.append(np.fromstring(chunk.ix[2][0], sep=' '))
        i = i + 1
    real = np.array(real)
    res = np.array(res)
    return ID, real, res

#=====================================================================
#=================INTERNAL-FUNCTIONS==================================
#=====================================================================
def MMean(real, res, i1, i2):
    """
    Internal Function
    """
    S = 0.0
    N = 0
    for i in range(len(real)):
        if real[i][i1] == 1:
            S = S + res[i][i2]
            N = N + 1
    return S / N


def MSD(real, res, i1, i2):
    """
    Internal Function
    """
    S = 0.0
    N = 0
    m = MMean(real, res, i1, i2)
    for i in range(len(real)):
        if real[i][i1] == 1:
            S = S + (res[i][i2] - m) ** 2
            N = N + 1
    return np.sqrt(S / N)


def Hist(real, res, i1, i2):
    S = []
    for i in range(len(real)):
        if real[i][i1] == 1:
            S.append(res[i][i2])
    S = np.array(S)
    w = np.ones_like(S) / len(S)
    return S, w


def NHU(Hlayers):
    """
    Count Number of Units in ALL Hidden Layers
    """
    s = np.zeros(len(Hlayers))
    i = 0
    for j in Hlayers:
        try:
            s[i] = int(j)
        except:
            for k in j:
                s[i] = s[i] + int(k)
        i = i + 1
    return s
#=====================================================================
#=================MATRIX-METRICS======================================
#=====================================================================


def WTA(ID, real, res):
    """
    Return Winner Takes All Matrix from ID,real,res
    """
    CM = np.zeros((4, 4), dtype=int)
    for i in range(len(ID)):
        CM[real[i].argmax(), res[i].argmax()] = CM[
            real[i].argmax(), res[i].argmax()] + 1
    return CM


def ConfusionMatrix(real, res, TP=0.5, FP=0.5):
    """
    Return [0]Numpy Array CM,[1]Numpy Array NormCM,[2]Pandas Dataframe CM,[3]Pandas Dataframe NormCM  from ID,real,res
    """
    Ntypes = np.zeros((4, 1))
    CM = np.zeros((4, 4), dtype=int)
    for i in range(np.shape(real)[0]):
        p = real[i].argmax()
        Ntypes[p] = Ntypes[p] + 1

        if (res[i][p] >= TP):
            CM[p, p] = CM[p, p] + 1

        for n in range(4):
            if ((n != p) and (res[i][n] >= FP)):
                CM[p, n] = CM[p, n] + 1
    NCM = CM / Ntypes
    dfCM = pd.DataFrame.from_records(CM, index=['E', 'S', 'I', 'Q'], columns=[
                                     'E(unit)', 'S(unit)', 'I(unit)', 'Q(unit)'])
    dfNCM = pd.DataFrame.from_records(NCM, index=['E', 'S', 'I', 'Q'], columns=[
                                      'E(unit)', 'S(unit)', 'I(unit)', 'Q(unit)'])
    return CM, NCM, dfCM, dfNCM


def AverageMatrix(real, res):
    """
    Return [0]np.Array AverageMatrix,[1]npArray SDMatrix,[2]Pandas Dataframe AverageMatrix,[3]Pandas Dataframe SDMatrix  from ID,real,res
    """
    MM = np.zeros((4, 4))
    DM = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            MM[i, j] = MMean(real, res, i, j)
            DM[i, j] = MSD(real, res, i, j)
    dfMM = pd.DataFrame.from_records(MM, index=['E', 'S', 'I', 'Q'], columns=[
                                     'E(unit)', 'S(unit)', 'I(unit)', 'Q(unit)'])
    dfDM = pd.DataFrame.from_records(DM, index=['E', 'S', 'I', 'Q'], columns=[
                                     'E(unit)', 'S(unit)', 'I(unit)', 'Q(unit)'])
    return MM, DM, dfMM, dfDM

#====================================================================
#================FOLDER-STATISTICS===================================
#====================================================================


def Folder_Classification_Report(Folder, TP=0.5, FP=0.5):
    n = []
    d = []
    Hlayers = []
    TPE = []
    TPS = []
    TPI = []
    TPQ = []
    accuracy = []
    filelist = [f for f in listdir(Folder) if f.endswith('res')]
    print 'Reading %d files in %s' % (len(filelist), Folder)
    for i, f in enumerate(filelist):
        datafile = 'Cres/' + f
        ID, real, res = ReadData(datafile)
        CM, NCM, dfM, dfN = ConfusionMatrix(real, res, TP, FP)
        #MM,DM,dfMM,dfDM = funcs.AverageMatrix(real,res)

        d.append(float(f[f.find('d') + 1:f.find('c')]))
        n.append(float(f[f.find('n') + 1:f.find('d')]))
        Hlayers.append([f[j + 1:j + 3]
                        for j in xrange(len(f)) if f.find('h', j) == j])
        TPE.append(NCM[0, 0])
        TPS.append(NCM[1, 1])
        TPI.append(NCM[2, 2])
        TPQ.append(NCM[3, 3])
        accuracy.append((TPE[i] + TPS[i] + TPI[i] + TPQ[i]) / NCM.sum())
        print i, f, accuracy[i]

    accuracy = np.array(accuracy)
    d = np.array(d)
    n = np.array(n)
    NL = np.array([len(j) for j in Hlayers])
    TPE = np.array(TPE)
    TPS = np.array(TPS)
    TPI = np.array(TPI)
    TPQ = np.array(TPQ)
    DF = pd.DataFrame({'File': filelist, 'd': d, 'n': n, 'Accuracy': accuracy,
                       'Layers': Hlayers, 'TPE': TPE, 'TPS': TPS, 'TPI': TPI, 'TPQ': TPQ})
    return DF


def Folder_Regression_Report(Folder):

    AVE_A = 0.42128425
    STD_A = 0.28714299
    AVE_Z = 0.09973424
    STD_Z = 0.05802179

    n = []
    d = []
    Hlayers = []
    sdA = []
    sdZ = []
    filelist = [f for f in listdir('AZres') if f.endswith('res')]
    del filelist[filelist.index('i96h24h12n0.02d0.1c1000o2R11.res')]
    print 'Reading %d files in %s' % (len(filelist), Folder)

    for i, f in enumerate(filelist):
        datafile = 'AZres/' + f
        ID, real, res = ReadData(datafile)

        uAn = res[:, 0]
        rAn = real[:, 0]
        uZn = res[:, 1]
        rZn = real[:, 1]

        dAn = uAn - rAn
        dZn = uZn - rZn

        uA = 3.0 * STD_A * uAn + AVE_A
        uZ = 3.0 * STD_Z * uZn + AVE_Z

        rA = 3.0 * STD_A * rAn + AVE_A
        rZ = 3.0 * STD_Z * rZn + AVE_Z

        dA = uA - rA
        dZ = uZ - rZ

        d.append(float(f[f.find('d') + 1:f.find('c')]))
        n.append(float(f[f.find('n') + 1:f.find('d')]))
        Hlayers.append([f[j + 1:j + 3]
                        for j in xrange(len(f)) if f.find('h', j) == j])
        sdA.append(dA.std(ddof=1.0))
        sdZ.append(dZ.std(ddof=1.0))
        print i, f, sdA[i], sdZ[i]
        i = i + 1

    d = np.array(d)
    n = np.array(n)
    NL = np.array([len(j) for j in Hlayers])
    sdA = np.array(sdA)
    sdZ = np.array(sdZ)
    DF = pd.DataFrame(
        {'File': filelist, 'd': d, 'n': n, 'SDA': sdA, 'SDZ': sdZ, 'Layers': Hlayers})
    return DF


#====================================================================
#=================PLOT-FUNCTIONS=====================================
#====================================================================
    #fig = plt.figure(figsize=(5,5))
    #ax = fig.add_subplot(111)
    # ax.grid(False)
    #ax.set_xticklabels(['E', 'S', 'I', 'Q'] )
    #ax.set_yticklabels(['E', 'S', 'I', 'Q'] )
    # ax.matshow(df,cmap=plt.cm.GnBu)
    # for (i, j), z in np.ndenumerate(df):
    #    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    # plt.tight_layout()

def plot_reg(datafile):
    AVE_A = 0.42128425
    STD_A = 0.28714299
    AVE_Z = 0.09973424
    STD_Z = 0.05802179

    ID, real, res = ReadData(datafile)

    uAn = res[:, 0]
    rAn = real[:, 0]
    uZn = res[:, 1]
    rZn = real[:, 1]

    dAn = uAn - rAn
    dZn = uZn - rZn

    uA = 3.0 * STD_A * uAn + AVE_A
    uZ = 3.0 * STD_Z * uZn + AVE_Z

    rA = 3.0 * STD_A * rAn + AVE_A
    rZ = 3.0 * STD_Z * rZn + AVE_Z

    dA = uA - rA
    dZ = uZ - rZ

    fig = plt.figure(figsize=(20,12))
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))

    ax1.set_title('Extinction Standard Deviation Histogram')
    ax1.hist(dA,bins=10)
    ax2.set_title('Redshift Standard Deviation Histogram')
    ax2.hist(dZ,bins=10)

    ax3.set_title('Extiction Data')
    ax3.set_xlabel('Unit')
    ax3.set_ylabel('True')
    ax3.scatter(uA,rA,alpha=0.7)
    ax3.plot(np.linspace(1.1*uA.min(),1.1*uA.max(),100),np.linspace(1.1*uA.min(),1.1*uA.max(),100),'r')

    ax4.set_title('Redshift Data')
    ax4.set_xlabel('Unit')
    ax4.set_ylabel('True')
    ax4.scatter(uZ,rZ)
    ax4.plot(np.linspace(1.1*uZ.min(),1.1*uZ.max(),100),np.linspace(1.1*uZ.min(),1.1*uZ.max(),100),'r')


def plot_reg_hist(datafile):
    AVE_A = 0.42128425
    STD_A = 0.28714299
    AVE_Z = 0.09973424
    STD_Z = 0.05802179

    ID, real, res = ReadData(datafile)

    uAn = res[:, 0]
    rAn = real[:, 0]
    uZn = res[:, 1]
    rZn = real[:, 1]

    dAn = uAn - rAn
    dZn = uZn - rZn

    uA = 3.0 * STD_A * uAn + AVE_A
    uZ = 3.0 * STD_Z * uZn + AVE_Z

    rA = 3.0 * STD_A * rAn + AVE_A
    rZ = 3.0 * STD_Z * rZn + AVE_Z

    dA = uA - rA
    dZ = uZ - rZ

    fig = plt.figure(figsize=(17,8))
    ax1 = plt.subplot2grid((1,2), (0,0))
    ax2 = plt.subplot2grid((1,2), (0,1))
    ax1.set_title('Extinction SD Hist')
    ax1.hist(dA)
    ax2.set_title('Redshift SD Hist')
    ax2.hist(dZ)
    #plt.save('Ext-Red Histograms.png')

def plot_reg_scat(datafile):
    AVE_A = 0.42128425
    STD_A = 0.28714299
    AVE_Z = 0.09973424
    STD_Z = 0.05802179

    ID, real, res = ReadData(datafile)

    uAn = res[:, 0]
    rAn = real[:, 0]
    uZn = res[:, 1]
    rZn = real[:, 1]

    dAn = uAn - rAn
    dZn = uZn - rZn

    uA = 3.0 * STD_A * uAn + AVE_A
    uZ = 3.0 * STD_Z * uZn + AVE_Z

    rA = 3.0 * STD_A * rAn + AVE_A
    rZ = 3.0 * STD_Z * rZn + AVE_Z

    dA = uA - rA
    dZ = uZ - rZ

    fig = plt.figure(figsize=(17,8))
    ax1 = plt.subplot2grid((1,2), (0,0))
    ax2 = plt.subplot2grid((1,2), (0,1))
    ax1.set_title('Extiction')
    ax1.set_xlabel('Unit')
    ax1.set_ylabel('True')
    ax1.scatter(uA,rA,alpha=0.7)
    ax1.plot(np.linspace(uA.min(),uA.max(),100),np.linspace(uA.min(),uA.max(),100),'r')
    ax2.set_title('Redshift')
    ax2.set_xlabel('Unit')
    ax2.set_ylabel('True')
    ax2.scatter(uZ,rZ)
    ax2.plot(np.linspace(uZ.min(),uZ.max(),100),np.linspace(uZ.min(),uZ.max(),100),'r')
    #plt.save('Ext-Red Linear.png')
def make_box(xx, yy):
    """
    Make Boxplot (xarray, yarray)
    """
    db = []
    for uni in np.unique(xx):
        db.append(yy[np.where(xx == uni)])
    plt.boxplot(db)
    plt.xticks(range(1, len(np.unique(xx)) + 1), np.unique(xx))
    return db


def plot_CM(df, datafile, TP, FP):
    fig = plt.figure(figsize=(9, 7))
    ax = seaborn.heatmap(df, annot=True)
    ax.set_title("Confusion matrix (TP> %0.1f, DP> %0.1f) for file: %s" % (
        TP, FP, datafile), fontsize=12)
    #plt.save('Confusion matrix.png')

def plot_MM(dfM, dfsd, datafile):
    fig = plt.figure(figsize=(9, 7))
    ax = seaborn.heatmap(dfM, annot=True)
    ax.set_title("Average for file: %s" % (datafile), fontsize=12)
    # for (i, j), z in np.ndenumerate(dfM):
    #    ax.text(j, i, '{:0.2f} +/- {:0.2f}'.format(z,dfsd.ix[i,j]), ha='center', va='center')
    # plt.tight_layout()
    #plt.save('Average matrix.png')

def plot_CM_MM(dfcm, dfM, dfsd, datafile, TP, FP):
    fig = plt.figure(figsize=(16, 10))
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax1.grid(False)
    ax1.set_xticklabels(['', 'E', 'S', 'I', 'Q'])
    ax1.set_yticklabels(['', 'E', 'S', 'I', 'Q'])
    ax1.matshow(dfcm, cmap=plt.cm.GnBu)
    ax1.set_title("Confusion matrix (TP> %0.1f, DP> %0.1f) for file: %s" % (
        TP, FP, datafile), fontsize=12)
    for (i, j), z in np.ndenumerate(dfcm):
        ax1.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    ax2 = plt.subplot2grid((1, 3), (0, 2))
    ax2.grid(False)
    ax2.set_xticklabels(['', 'E', 'S', 'I', 'Q'])
    ax2.set_yticklabels(['', 'E', 'S', 'I', 'Q'])
    ax2.matshow(dfM, cmap=plt.cm.GnBu)
    ax2.set_title("Average Matrix for file: %s" % (datafile), fontsize=12)
    for (i, j), z in np.ndenumerate(dfM):
        ax2.text(
            j, i, '{:0.2f} +/- {:0.2f}'.format(z, dfsd.ix[i, j]), ha='center', va='center')
    plt.tight_layout()


def plot_bars(df):
    fig = plt.figure(figsize=(19, 7))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    #ax.set_title("BarPlot for file: %s"%datafile,fontsize=12)
    df.plot(kind='bar', ax=ax1)
    df.plot(kind='bar', stacked='true', ax=ax2)
    plt.tight_layout()
    #plt.save('Barplot.png')


def make_hists(real, res):
    X = 4
    Y = 4
    fig = plt.figure(figsize=(16, 8))
    fig.text(-0.035, 0.125, 'Histograms For \n Galaxy Type Q', ha="center",
             va="center", size=12, color='white',bbox=dict(boxstyle="square", alpha=0.75))
    fig.text(-0.035, 0.375, 'Histograms For \n Galaxy Type I', ha="center",
             va="center", size=12, color='white',bbox=dict(boxstyle="square", alpha=0.75))
    fig.text(-0.035, 0.625, 'Histograms For \n Galaxy Type S', ha="center",
             va="center", size=12, color='white',bbox=dict(boxstyle="square", alpha=0.75))
    fig.text(-0.035, 0.875, 'Histograms For \n Galaxy Type E', ha="center",
             va="center", size=12, color='white', bbox=dict(boxstyle="square", alpha=0.75))

    fig.text(0.125, 1.025, 'E - Unit', rotation=270, ha="center",
             va="center", size=12, color='white',bbox=dict(boxstyle="rarrow", alpha=0.75))
    fig.text(0.375, 1.025, 'S - Unit', rotation=270, ha="center",
             va="center", size=12, color='white',bbox=dict(boxstyle="rarrow,pad=0.3", alpha=0.75))
    fig.text(0.625, 1.025, 'I - Unit', rotation=270, ha="center",
             va="center", size=12, color='white',bbox=dict(boxstyle="rarrow,pad=0.3", alpha=0.75))
    fig.text(0.875, 1.025, 'Q - Unit', rotation=270, ha="center",
             va="center", size=12, color='white',bbox=dict(boxstyle="rarrow,pad=0.3", alpha=0.75))

    ax1 = plt.subplot2grid((X, Y), (0, 0))
    ax2 = plt.subplot2grid((X, Y), (0, 1))
    ax3 = plt.subplot2grid((X, Y), (0, 2))
    ax4 = plt.subplot2grid((X, Y), (0, 3))

    ax5 = plt.subplot2grid((X, Y), (1, 0))
    ax6 = plt.subplot2grid((X, Y), (1, 1))
    ax7 = plt.subplot2grid((X, Y), (1, 2))
    ax8 = plt.subplot2grid((X, Y), (1, 3))

    ax9 = plt.subplot2grid((X, Y), (2, 0))
    ax10 = plt.subplot2grid((X, Y), (2, 1))
    ax11 = plt.subplot2grid((X, Y), (2, 2))
    ax12 = plt.subplot2grid((X, Y), (2, 3))

    ax13 = plt.subplot2grid((X, Y), (3, 0))
    ax14 = plt.subplot2grid((X, Y), (3, 1))
    ax15 = plt.subplot2grid((X, Y), (3, 2))
    ax16 = plt.subplot2grid((X, Y), (3, 3))

    h, w = Hist(real, res, 0, 0)
    ax1.hist(h, weights=w)
    ax1.set_ylabel('E (TP)')

    h, w = Hist(real, res, 0, 1)
    ax2.hist(h, weights=w)
    ax2.set_ylabel('S')

    h, w = Hist(real, res, 0, 2)
    ax3.hist(h, weights=w)
    ax3.set_ylabel('I')

    h, w = Hist(real, res, 0, 3)
    ax4.hist(h, weights=w)
    ax4.set_ylabel('Q')

    h, w = Hist(real, res, 1, 0)
    ax5.hist(h, weights=w)
    ax5.set_ylabel('E')

    h, w = Hist(real, res, 1, 1)
    ax6.hist(h, weights=w)
    ax6.set_ylabel('S (TP)')

    h, w = Hist(real, res, 1, 2)
    ax7.hist(h, weights=w)
    ax7.set_ylabel('I')

    h, w = Hist(real, res, 1, 3)
    ax8.hist(h, weights=w)
    ax8.set_ylabel('Q')

    h, w = Hist(real, res, 2, 0)
    ax9.hist(h, weights=w)
    ax9.set_ylabel('E')

    h, w = Hist(real, res, 2, 1)
    ax10.hist(h, weights=w)
    ax10.set_ylabel('S')

    h, w = Hist(real, res, 2, 2)
    ax11.hist(h, weights=w)
    ax11.set_ylabel('I (TP)')

    h, w = Hist(real, res, 2, 3)
    ax12.hist(h, weights=w)
    ax12.set_ylabel('Q')

    h, w = Hist(real, res, 3, 0)
    ax13.hist(h, weights=w)
    ax13.set_ylabel('E')

    h, w = Hist(real, res, 3, 1)
    ax14.hist(h, weights=w)
    ax14.set_ylabel('S')

    h, w = Hist(real, res, 3, 2)
    ax15.hist(h, weights=w)
    ax15.set_ylabel('I')

    h, w = Hist(real, res, 3, 3)
    ax16.hist(h, weights=w)
    ax16.set_ylabel('Q (TP)')
    plt.tight_layout()
# plt.savefig('hist.pdf')

#=====================================================================
#=================MAIN-REPORTS==========================================
#=====================================================================


def Full_Clasification_Report(datafile):
    ID, real, res = ReadData(datafile)
    #wtaM = WTA(ID, real, res)

    MM, DM, dfMM, dfDM = AverageMatrix(real, res)
    plot_MM(dfMM,dfDM,datafile)

    CM, NCM, dfM, dfN = ConfusionMatrix(real, res, TP=0.5, FP=0.5)
    plot_CM(dfN, datafile, TP=0.5, FP=0.5)

    CM, NCM, dfM, dfN = ConfusionMatrix(real, res, TP=0.9, FP=0.9)
    plot_CM(dfN, datafile, TP=0.9, FP=0.9)

    plot_bars(dfN)
    make_hists(real, res)

def Full_Regression_Report(datafile):
    plot_reg(datafile)
