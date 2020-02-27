from neededModules import * #contains imports and constants
from gradient import * #calculates gradient of network
from Intializer import * #loads data and randomizes starting weights
from WorkbooksTests import saveData

#fills a np array of the specificied size with random values 0<x<1
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


(x_train,y_train),(x_test,y_test) = loadStartingData()

W1,B1,W2,B2 = createWeightsAndBiases()

chosenEpochs = []
chosenCosts = []
#essentially a while true with a built in exit
for epoch in range(80000):
    W1temp = np.zeros((layer1Len,layer2Len)) #L1 by L2 matrix
    B1temp =  np.zeros(layer2Len).flatten() #L2 by 1 vector

    W2temp = np.zeros((layer2Len,layer3Len)) #L2 by L3 matrix
    B2temp =  np.zeros(layer3Len).flatten() # L3 by 1 vector

    cost = 0
    gradChangeSum = 0

    #takes in one input point (e.g. '3') and calculates outputs
    for idx in range(numTrained):
        idx = random.randint(1,60000)-1
        Z1 = calcZVals(W1,x_train[idx],B1)
        A2 = sigmoidActivation(Z1)

        Z2 = calcZVals(W2,A2,B2)
        A3 = softmaxActivation(Z2)

        #idx 0 = W1grad, 1 = B1grad, 2 = W2grad, 3 = B2grad
        gradientChange = gradientCalculations(x_train[idx],y_train[idx],A2,A3,W1,B1,W2,B2)
        gradChangeSum = np.add(gradChangeSum,gradientChange)

        cost += getCost(A3,y_train[idx])

    gradMomentum = gradMomentum *momentum + (1-momentum)*gradChangeSum

    W1 = np.add(W1,gradMomentum[0]*learn)
    B1 = np.add(B1,gradMomentum[1]*learn)
    W2 = np.add(W2,gradMomentum[2]*learn)
    B2 = np.add(B2,gradMomentum[3]*learn)

    cost = cost / numTrained
    print(epoch, cost)

    #saves data for review after the run
    if epoch % 1000 == 0:
        chosenEpochs.append(epoch)
        chosenCosts.append(cost)
        saveData(epoch,cost)

    if keyboard.is_pressed('-'):
        keepingOn = not keepingOn

    if keepingOn:
        keyboard.press('=')

    if keyboard.is_pressed('+') or cost < 0.005:
        break


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

plt.plot(chosenEpochs,chosenCosts)
titleString = 'Cost vs Epochs for momentum: ' + str(momentum)
plt.title(titleString)
plt.savefig('Epoch_vs_Cost.png')
