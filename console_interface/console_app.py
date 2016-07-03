from utils import utils
from learning_logic import supervised, decision_tree
import multiprocessing
import pkg_resources, os
import csv
import random


def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results!=None:
        print str(tree.results)
    else:
        # Print the criteria
        print 'Column ' + str(tree.column)+' : '+str(tree.value)+'? '

        # Print the branches
        print indent+'True->',
        printtree(tree.trueNodes,indent+'  ')
        print indent+'False->',
        printtree(tree.falseNodes,indent+'  ')


# Removes every row that has blank data in order not to bias the algorithm
def normalizeDataset(dataSet):
    rowLength = len(dataSet[0])
    dataSetLength = len(dataSet)
    lineIndex = 0
    for lineNumber in range(0, dataSetLength):
        line = dataSet[lineIndex]
        deletedLine = False
        for col in range(0 , rowLength):
            if(line[col]==''):
                del dataSet[lineIndex]
                deletedLine = True
            if deletedLine: break
        if not deletedLine: lineIndex += 1
    return dataSet

# Removes entire column from the dataset by index, columnIndex starts at 0
def removeColumn(dataSet, columnIndex):
    for row in dataSet:
        del row[columnIndex]


def printDataSet(dataSet):
    print "DATASET:"
    print ''
    for line in dataSet:
        print line

# Get average value of a column, columnIndex starts at 0
def average(dataSet, columnIndex):
    sum = 0
    for row in dataSet:
        sum += row[columnIndex]
    return sum / len(dataSet)

# Get max value of a column, columnIndex starts at 0
def maxValue(dataset, columnIndex):
    return max([row[columnIndex] for row in dataset])

# Get min value of a column, columnIndex starts at 0
def minValue(dataset, columnIndex):
    return min([row[columnIndex] for row in dataset])

# Splits a dataset randomly into a number of datasets
def randomSplit(dataSet, subsetQuantity):
    startingIndex = 0
    dataSets = []
    subsetLength = int(round(len(dataSet) / subsetQuantity))
    random.shuffle(dataSet)
    for i in range(0, subsetQuantity):
        dataSets.append( dataSet[startingIndex : startingIndex+subsetLength] )
        startingIndex += subsetLength
    return dataSets

def decisionTreeMain():
    resource_package = 'resources'
    filename = 'titanic.train.csv'
    resource_path = os.path.join('training_data', filename)
    filePath = pkg_resources.resource_filename(resource_package, resource_path) # Gets path of file from another package

    lines = csv.reader(open(filePath, "rb"))
    dataset = list(lines)
    headers = dataset[0]
    del dataset[0] # removes headers from dataset
    dataset = normalizeDataset(dataset)
    printDataSet(dataset)
    del dataset[0] # test
    print('-------------')
    print('***** Splits randomly subsets in order to create a random forest ****')
    subsets = randomSplit(dataset, 3)
    printDataSet(subsets)
    #removeColumn(dataset, 3)
    #printDataSet(dataset)

    #datasetEntropy = decision_tree.entropy(my_data) # 2.40
    #print('Entropy in {0} dataset: {1}').format(filename, str(datasetEntropy))
    decision_tree.postponeColumn(dataset, 2) # Shifts 'Survive' column to the last
    rowToClassify = dataset[1]
    del dataset[1]

    tree = decision_tree.buildTreeWithHeigth(dataset, maxHeigth=5)
    printtree(tree)
    print "Decidimos la siguiente fila:"
    print rowToClassify
    print "Resultado -> " + str(decision_tree.classifyInTree(tree, rowToClassify))





decisionTreeMain()
