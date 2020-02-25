import tensorflow as tf
import numpy as np
import random
from gradient import *
import keyboard
import matplotlib.pyplot as plt
from WorkbooksTest import saveData

#fills a np array of the specificied size with random values 0<x<1
def fillWithRands(dim1,dim2=1):
    temp = np.zeros((dim1,dim2))
    for i in range(dim1):
        for j in range(dim2):
            temp[i,j] = random.randint(-100,100)/100
    return temp

def calcZVals(Weights,Actives,Biases):
    temp = np.dot(Actives,Weights) #multiply
    temp = temp + Biases
    return temp

def sigmoidActivation(Zs):
    A = 1 /(1 + np.exp(-1*Zs))
    return A

def softmaxActivation(Zs):
    exps = np.exp(Zs)
    sum = np.sum(exps)
    return exps/sum

def getCost(A,labels): #squared error
    diff = A-labels
    squared = np.multiply(diff,diff) #squares differences
    squaredSum = np.sum(squared)/2
    return squaredSum

keepingOn = False
numLayers = 3
numCategories = 10

numTrained = 400

numTested = 4000

layer1Len = 784
layer2Len = 28
layer3Len = 10
learn = -0.13
learn = learn/numTrained

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test),784))

#converts y_train from labels to list of probabilities
y_temp = np.zeros((len(y_train),numCategories))
for idx in range(len(y_train)):
    y_temp[idx,y_train[idx]] = 1
y_train = y_temp

#converts y_test from labels to list of probabilities
y_temp = np.zeros((len(y_test),numCategories))
for idx in range(len(y_test)):
    y_temp[idx,y_test[idx]] = 1
y_test = y_temp

#creates random weights and biases
W1 = fillWithRands(layer1Len,layer2Len) #L1 by L2 matrix
B1 = fillWithRands(layer2Len).flatten() #L2 by 1 vector

W2 = fillWithRands(layer2Len,layer3Len) #L2 by L3 matrix
B2 = fillWithRands(layer3Len).flatten() # L3 by 1 vector

chosenEpochs = []
chosenCosts = []
#essentially a while true with a built in exit
for epoch in range(100000):
    W1temp = np.zeros((layer1Len,layer2Len)) #L1 by L2 matrix
    B1temp =  np.zeros(layer2Len).flatten() #L2 by 1 vector

    W2temp = np.zeros((layer2Len,layer3Len)) #L2 by L3 matrix
    B2temp =  np.zeros(layer3Len).flatten() # L3 by 1 vector

    cost = 0

    #takes in one input point (e.g. '3') and calculates outputs
    for idx in range(numTrained):
        Z1 = calcZVals(W1,x_train[idx],B1)
        A2 = sigmoidActivation(Z1)

        Z2 = calcZVals(W2,A2,B2)
        A3 = softmaxActivation(Z2)

        #idx 0 = W1grad, 1 = B1grad, 2 = W2grad, 3 = B2grad
        gradientChange = gradientCalculations(x_train[idx],y_train[idx],A2,A3,W1,B1,W2,B2)
        W1temp = np.add(gradientChange[0],W1temp)
        B1temp = np.add(gradientChange[1],B1temp)
        W2temp = np.add(gradientChange[2],W2temp)
        B2temp = np.add(gradientChange[3],B2temp)

        cost += getCost(A3,y_train[idx])

    W1 = np.add(W1,W1temp*learn)
    B1 = np.add(B1,B1temp*learn)
    W2 = np.add(W2,W2temp*learn)
    B2 = np.add(B2,B2temp*learn)

    cost = cost / numTrained
    print(epoch, cost)

    #saves data for review after the run
    if epoch % 100 == 0:
        chosenEpochs.append(epoch)
        chosenCosts.append(cost)
        saveData(epoch,cost)

    if keyboard.is_pressed('s'):
        keepingOn = not keepingOn

    if keepingOn:
        keyboard.press('w')

    if keyboard.is_pressed('e') or cost < 0.05:
        break


plt.plot(chosenEpochs,chosenCosts)
plt.savefig('Epoch_vs_Cost.png')


tested = 0
right = 0
for idx in range(numTested):
    Z1 = calcZVals(W1,x_test[idx],B1)
    A2 = sigmoidActivation(Z1)

    Z2 = calcZVals(W2,A2,B2)
    A3 = softmaxActivation(Z2)

    tested += 1
    if np.argmax(A3) == np.argmax(y_test[idx]):
        right += 1

print(right/tested)
