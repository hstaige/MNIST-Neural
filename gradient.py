from neededModules import *

def createJacobian(A3):
    """
    Creates a jacobian matrix used to calculate B2 gradient. Represents how
    changing one input value to the last layer changes the output activations.

    INPUT: A3, [layer3len,1]

    OUTPUT: Jacobian matrix, [layer3len,layer3len]
    """
    jacob = np.zeros((layer3Len,layer3Len))
    for i in range(layer3Len):
        for j in range(layer3Len):
            if i == j: #how input changes its own activation
                jacob[i,j] = A3[i]*(1-A3[j])
            else: #how input changes anothers activation
                jacob[i,j] = -1*A3[i]*A3[j]
    return jacob


def sigmoidGradient(A):
    """
    Calculates the gradient of the activations of the sigmoid activated layers
    with respect to their input.

    INPUT: A, [layerlen, 1]

    OUTPUT: sigmoid gradient, [layerlen, 1]
    """
    gradient = np.multiply(A,(-A+1))
    return gradient


def calcB2grad(y_train,A3):
    """
    Calculates the gradient of the cost with respect to B2.

    INPUT: y_train, [layer3len,1] ; A3, [layer3len,1]

    OUTPUT: B2grad, [layer3len,1]
    """
    costGrad = A3-y_train #numCategories by 1 vector
    B2Jacobian = createJacobian(A3) #A3 by A3 matrix
    B2grad = np.dot(costGrad.transpose(),B2Jacobian) #how changing something in b2
    #affects the cost function:
    return B2grad


def calcW2grad(B2Grad, A2):
    """
    Calculates the gradient of the cost with respect to W2

    INPUT: B2Grad, [layer3len, 1] ; A2, [layer2len, 1]

    OUTPUT: W2grad [layer3len, layer2Len]
    """
    temp1 = B2Grad.reshape((layer3Len,1))
    temp2 = A2.reshape((1,layer2Len))
    W2grad = np.dot(temp1, temp2 )
    return W2grad


def calcB1grad(B2grad,W2,A2):
    """
    Calculates the gradient of the cost with respect to B1 using the chain rule

    INPUT: B2grad, [layer3Len,1] ; W2, [layer2Len, layer3Len] ;
           A2, [layer2len, 1]

    OUTPUT: B1grad, [layer2Len, 1]
    """
    temp1 = np.dot(W2,B2grad) #layer2Len * 1 vector
    sigmGradient = sigmoidGradient(A2) #layer2len * 1 vector
    B1grad = np.multiply(sigmGradient,temp1)
    return B1grad


def calcW1Grad(B1grad, x_train):
    """
    Calculates the gradient of the cost with respect to W1 using the chain rule

    INPUT: B1grad, [layer2Len,1] ; x_train, [layer1Len, 1] ;

    OUTPUT: W1grad, [layer1Len, layer2Len]
    """

    temp1 = B1grad.reshape((layer2Len,1))
    temp2 = x_train.reshape((1,layer1Len))
    W1grad = np.dot(temp1,temp2)
    return W1grad

#handles calculation of gradient for NN
def gradientCalculations(x_train,y_train,A2,A3,W1,B1,W2,B2):
    B2grad = calcB2grad(y_train,A3) #numCategories by 1 vector
    W2grad = calcW2grad(B2grad,A2) #NumCategories by layer2Len matrix
    B1grad = calcB1grad(B2grad,W2,A2) #layer2Len by 1 vector
    W1grad = calcW1Grad(B1grad, x_train)

    return [W1grad.transpose(), B1grad.transpose(), W2grad.transpose(), B2grad.transpose()]
    #note, I dont know why i have to transpose them but it works `\(0_0)/`
