# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:40:10 2016

@author: a.kuester
"""

#%% AB 3
import numpy as np
import nnwplot
import matplotlib.pyplot as plt

#%% ErrorRate
def ErrorRate(Y,T):
    if Y.ndim==1 or Y.shape[0]==1:
        errors=Y!=T
        return errors.sum()/Y.size
    else: # für mehrere Ausgaben in one-hot Kodierung:
        errors=Y.argmax(0)!=T.argmax(0)
        return errors.sum()/Y.shape[1]
        
#%%
class SLN:
    def __init__(self,dIn,cOut): # Konstruktor
        self._features=dIn
        if dIn ==2:
            self._print=True
        else:
            self._print=False
        self._b=np.zeros(cOut)[np.newaxis].T
        np.random.seed(42)
        self._W=np.random.randn(cOut,dIn)/np.sqrt(dIn)
        if cOut==1:
            self.neuron=self.threshold
        else:
            self.neuron=self.thresholdMult
    def netsum(self,X):
         return self._W.dot(X)+self._b        
    def threshold(self,X):
        return self.netsum(X)>=0
    def onehot(self,T):
        e=np.identity(self._W.shape[0])
        return e[:,T.astype(int)]
    def thresholdMult(self,X):
        return self.onehot(self.netsum(X).argmax(0))
    def DeltaTrain(self, X, T, eta, maxIter, maxErrorRate):
        best = self;
        bestError = 2;
        bestIt = 0;
        N=X.shape[1]    # Anzahl Trainingsdaten
        x0 = np.ones(N)[np.newaxis]
        plt.ion() # interactive mode on
        for it in range(maxIter):
            Y = self.neuron(X)
            err = ErrorRate(Y, T)
            #if (it%20) == 0:
                #print('#{} {} {} {}'.format(it,self._W,self._b,err))
                #nnwplot.plotTwoFeatures(X,T,self.neuron)
                #plt.pause(0.05) # warte auf GUI event loop
            if err<bestError:
                bestError = err
                best = self
                bestIt = it
            if err <= maxErrorRate:
                break
            self._W+=eta*(T-Y).dot(X.T)/N
            self._b+=eta*(T-Y).dot(x0.T)/N
        self=best
        if self._print:
            #print('#{} {} {} {}'.format(it,self._W,self._b,err))
            nnwplot.plotTwoFeatures(X,T,self.neuron)
            plt.pause(0.05) # warte auf GUI event loop
        else:
            print('Training with {} features finished'.format(self._features))
    
        return bestError, bestIt
    def test(self,X):
        r= slnIris.neuron(X)
        iris_type=np.argmax(r)
        print('Input:')
        print('{} '.format(X))
        print('Iris-Type: {}'.format(iris_type))
        

#%% Iris-Daten Laden
iris = np.loadtxt("iris.csv",delimiter=',')
X=iris[:,0:4].T
T=iris[:,4]

#%% oneHotTest
TestT = np.array([0,2,1,2])
slnTest = SLN(2,3)
TestResult=slnTest.onehot(TestT)

#%% Training mit 3 Iris-Blütenarten 
# merkemale 1,2
print('merkemale 1,2')
plt.figure()
newX=X[:2,:]
slnIris1 = SLN(2,3)
oneHot1=slnIris1.onehot(T)
slnIris1.DeltaTrain(newX,oneHot1,0.1,3000,0.01)

# merkemale 2,3
print('merkemale 2,3')
plt.figure()
newX=X[1:3,:]
slnIris2 = SLN(2,3)
oneHot2=slnIris2.onehot(T)
slnIris2.DeltaTrain(newX,oneHot2,0.1,3000,0.01)

# merkemale 3,4
print('merkemale 3,4')
plt.figure()
newX=X[2:,:]
slnIris3 = SLN(2,3)
oneHot3=slnIris3.onehot(T)
slnIris3.DeltaTrain(newX,oneHot3,0.1,3000,0.01)


#%% Training mit 3 Iris-Blütenarten
# alle merkmale
plt.figure()
newX=X[:,:]
slnIris4 = SLN(4,3)
oneHot4=slnIris4.onehot(T)
slnIris4.DeltaTrain(newX,oneHot4,0.1,100000,0.01)
finalResult=slnIris4.neuron(newX)

#Trainiertes netz testen
testX = np.array([[5.2],[1.2],[5.1],[3.4]])
slnIris4.test(testX)


