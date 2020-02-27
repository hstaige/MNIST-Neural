from neededModules import *

def fillWithRands(dim1,dim2=1):
    temp = np.zeros((dim1,dim2))
    for i in range(dim1):
        for j in range(dim2):
            temp[i,j] = random.randint(-100,100)/100
    return temp

def loadStartingData():
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

    return (x_train,y_train),(x_test,y_test)
#creates random weights and biases
def createWeightsAndBiases():
    W1 = fillWithRands(layer1Len,layer2Len) #L1 by L2 matrix
    B1 = fillWithRands(layer2Len).flatten() #L2 by 1 vector

    W2 = fillWithRands(layer2Len,layer3Len) #L2 by L3 matrix
    B2 = fillWithRands(layer3Len).flatten() #L3 by 1 vector

    return W1,B1,W2,B2
