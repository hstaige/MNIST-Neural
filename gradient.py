import numpy as np

layer3Len = 10
layer2Len = 28

def createJacobian(A3):
    jacob = np.zeros((layer3Len,layer3Len))
    for i in range(layer3Len):
        for j in range(layer3Len):
            if i == j: #how input changes its own activation
                jacob[i,j] = A3[i]*(1-A3[j])
            else: #how input changes anothers activation
                jacob[i,j] = -1*A3[i]*A3[j]
    return jacob


def sigmoidGradient(A):
    gradient = np.multiply(A,(-A+1))
    # if np.amax(gradient) == 0:
    #     A = A - 0.001
    #     gradient = np.multiply(A,(-A+1))
    #     return gradient
    # else:
    return gradient


def calcB2grad(y_train,A3):
    costGrad = A3-y_train #numCategories by 1 vector
    B2Jacobian = createJacobian(A3) #A3 by A3 matrix
    B2grad = np.dot(costGrad.transpose(),B2Jacobian) #how changing something in b2
    #affects the cost function:
    return B2grad


def calcW2grad(B2Grad, A2):
    temp1 = B2Grad.reshape((10,1))
    temp2 = A2.reshape((1,28))
    W2grad = np.dot(temp1, temp2 )
    return W2grad


def calcB1grad(B2grad,W2,A2):
    temp1 = np.dot(W2,B2grad) #layer2Len * 1 vector
    sigmGradient = sigmoidGradient(A2) #layer2len * 1 vector
    B1grad = np.multiply(sigmGradient,temp1)
    return B1grad


def calcW1Grad(B1grad, x_train):
    temp1 = B1grad.reshape((28,1))
    temp2 = x_train.reshape((1,784))
    W1grad = np.dot(temp1,temp2)
    return W1grad

#handles calculation of gradient for NN
def gradientCalculations(x_train,y_train,A2,A3,W1,B1,W2,B2):
    B2grad = calcB2grad(y_train,A3) #numCategories by 1 vector
    W2grad = calcW2grad(B2grad,A2) #NumCategories by layer2Len matrix
    B1grad = calcB1grad(B2grad,W2,A2) #layer2Len by 1 vector
    W1grad = calcW1Grad(B1grad, x_train)

    return [W1grad.transpose(), B1grad.transpose(), W2grad.transpose(), B2grad.transpose()]
