#modules
import tensorflow as tf
import numpy as np
import random
import keyboard
import matplotlib.pyplot as plt
from openpyxl import load_workbook
import os

#constants
gradMomentum = 0 #initial value
keepingOn = False

rows = 1
cols = 1

numLayers = 3
numCategories = 10

numTrained = 40

numTested = 10000

layer1Len = 784
layer2Len = 200
layer3Len = 10

momentum = 0.8


learn = -0.13
learn = learn/numTrained

momentumL = []
accuracyL = []
