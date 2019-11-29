import csv
import random
import math
import operator
from random import seed
from random import randrange
import numpy as np

whichDataSet = 0

## Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Find the min and max values for each column

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        colvalues = [row[i] for row in dataset]
        min_value = min(colvalues) 
        max_value = max(colvalues)
        minmax.append([min_value, max_value])
    Normalize_Dataset(dataset, minmax)

# Normalize the dataset except last row for classification values
def Normalize_Dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] =float(row[i]) - (float(minmax[i][0])) / (float(minmax[i][1]) - float(minmax[i][0]))


### Splitting dataset methods ###

# Split a dataset into a train and test set
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    cross_validation_split(trainingSet, folds=10)

# Split a dataset into $k$ folds
def cross_validation_split(dataset, folds = 3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


####### Accuracy for classification problems ######

# Get accuracy of prediction #
def getAccuracy(actual,predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i][-1] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.00


##### Distances definition ######

#Euclidean Distance
def EuclideanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow(instance2[i]-instance1[i],2)
    return math.sqrt(distance)

def Polynomial_Kernel(instance1,instance2,length, p = 13):
    distance = 0
    a = np.array(instance1)
    b = np.asarray(instance2)
    a = a[0:np.size(a)-1:1]
    b = b[0:np.size(b)-1:1]
    a = np.array(a, dtype = float)
    b = np.array(b, dtype = float)
    
    l = np.power(1 + np.sqrt((np.dot(a, a)) - 2 * np.dot(a,b) + np.dot(b,b)),13)
    return l

def sigmoid(instance1, instance2, length):
    alpha = 1
    beta = 2
    f = length[0]
    l = length[-1]
    temp = alpha*((np.dot(np.array(instance1[f:l]), np.array(instance2[f:l])))) + beta
    return np.tanh(temp)

def RBF_kernel(instance1, instance2, length):
    sigma = 0.90
    a = np.array(instance1)
    b = np.asarray(instance2)
    a = a[0:np.size(a)-1:1]
    b = b[0:np.size(b)-1:1]
    a = np.array(a, dtype = float)
    b = np.array(b, dtype = float)
    l = 2 - 2*np.exp(-(np.power(sum(np.abs(a-b)), 2))/np.power(sigma, 2))
    return l


#Get neighbors
def getNeighbors(trainingSet, testInstance, num_neighbors, distancetype, *args):
    distances = []
    length = len(testInstance)-1
    for i in range(len(trainingSet)):
        if distancetype == "Euclidean" or "euclidean":
            dist = EuclideanDistance(testInstance, trainingSet[i], length)
        elif distancetype == "Polynomial":
            dist = Polynomial_Kernel(testInstance, trainingSet[i], length)
        elif distancetype == "Sigmoid" or "sigmoid":
            length = range(1,8,1) # For yeast dataset
            dist = sigmoid(testInstance, trainingSet[i], length)
        elif distancetype == "RBF":
            dist = RBF_kernel(testInstance, trainingSet[i], length)
        distances.append((trainingSet[i],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(num_neighbors):
        neighbors.append(distances[x][0])
    return neighbors

#Classification from neighbors (Classification problem)
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def main():
    filename = 'yeast.csv'
    dataset = load_csv(filename)
    
    # normalization of dataset
    dataset_minmax(dataset)

    # Splitting dataset between Training and Testing Set
    fold1 = cross_validation_split(dataset, folds=10)
    for i in range(10):
        testSet = fold1[i]
        training = []
        for j in range(10):
            if(i!=j):
                training += fold1[j]

    #generate predictions
    predictions = []
    num_neighbors = 3
    distancetype = input("distance type (Euclidean/Polynomial/Sigmoid/RBF) ")
    for i in range(len(testSet)):
        neighbors = getNeighbors(training, testSet[i], num_neighbors, distancetype)
        result = getResponse(neighbors)
        predictions.append(result)

    #Accuracy Assessment
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy :' + repr(accuracy) + '%')

main()
