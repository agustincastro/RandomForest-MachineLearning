from utils import utils
from learning_logic import supervised, decision_tree
import pkg_resources, os
import csv


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


# Removes every row that has blank data in order not to vias the algorithm
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

def removeColumn(dataSet, columnIndex):
    for row in dataSet:
        del row[columnIndex]



def printDataSet(dataSet):
    print "DATASET:"
    print ''
    for line in dataSet:
        print line

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


def bayesMain():
    resource_package = 'resources'
    resource_path = os.path.join('training_data', 'titanic.train.csv')
    filename = pkg_resources.resource_filename(resource_package, resource_path) # Gets path of file from another package

    data = pkg_resources.resource_string(resource_package, resource_path) # Gets data of file from another package

    splitRatio = 0.67
    # Loads the dataset from a CSV file
    dataset = utils.loadCsv(filename)

    # Processes dataset into a training set and test set
    trainingSet, testSet = utils.splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))

    # prepare model
    summaries = utils.summarizeByClass(trainingSet)
    # test model
    predictions = supervised.NaiveBayes(summaries, testSet)
    accuracy = supervised.getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)



decisionTreeMain()



