from utils import dataset as datasetModule
from learning_logic import decision_tree, random_forest
import pkg_resources, os
import csv


def maxElementsInNodeTest(columnToTestIndex = 11, numberOfTrees = 3, minNodes = 15):
    # ----- Import csv file -----
    resource_package = 'resources'
    filename = 'wine-quality-red.csv'
    resource_path = os.path.join('training_data', filename)
    filePath = pkg_resources.resource_filename(resource_package, resource_path) # Gets path of file from another package
    lines = csv.reader(open(filePath, "rb"))
    dataset = list(lines)

    #  ---- Prepare dataset for analysing  ------
    del dataset[0] # removes headers from dataset
    dataset = datasetModule.normalizeDataset(dataset)

    # ----- Separates testSet from Dataset ------
    testSet = datasetModule.getTestSet(dataset,columnToTestIndex)
    # Delete balanced testSet from original dataset
    dataset = [x for x in dataset if x not in testSet]

    #Split dataSet into different subsets in order to create decision trees
    subsets = datasetModule.randomSplit(dataset, numberOfTrees)

    # Creates random Forest
    print "***** Creating random forest with {} min nodes({}) trees *****".format(numberOfTrees, minNodes)
    variousTreesMultiprocessinng = random_forest.createDecisionTreesMultiprocessing(subsets, decision_tree.buildTreeWithMaxElementsInNode, minNodes = minNodes)

    # Classify testSet against the forest
    rightAnswersCount = 0
    for testRow in testSet:
        classificationResult = random_forest.classifyForestMultiprocessing(variousTreesMultiprocessinng, testRow)
        finalResult = random_forest.getFinalResult(classificationResult)
        if finalResult == testRow[columnToTestIndex]:
            rightAnswersCount += 1
    print "evaluated correctly {} out of {} tests".format(rightAnswersCount, len(testSet))
    print

def maxHeigthTest(columnToTestIndex = 11, numberOfTrees = 3, maxHeigth = 15):
    # ----- Import csv file -----
    resource_package = 'resources'
    filename = 'wine-quality-red.csv'
    resource_path = os.path.join('training_data', filename)
    filePath = pkg_resources.resource_filename(resource_package, resource_path) # Gets path of file from another package
    lines = csv.reader(open(filePath, "rb"))
    dataset = list(lines)

    #  ---- Prepare dataset for analysing  ------
    del dataset[0] # removes headers from dataset
    dataset = datasetModule.normalizeDataset(dataset)

    # ----- Separates testSet from Dataset ------
    testSet = datasetModule.getTestSet(dataset,columnToTestIndex)
    # Delete balanced testSet from original dataset
    dataset = [x for x in dataset if x not in testSet]

    #Split dataSet into different subsets in order to create decision trees
    subsets = datasetModule.randomSplit(dataset, numberOfTrees)

    # Creates random Forest
    print "***** Creating random forest with {} max heigth({}) trees. *****".format(numberOfTrees, maxHeigth)
    variousTreesMultiprocessinng = random_forest.createDecisionTreesMultiprocessing(subsets, decision_tree.buildTreeWithHeigth, maxHeigth = 15)

    # Classify testSet against the forest
    rightAnswersCount = 0
    for testRow in testSet:
        classificationResult = random_forest.classifyForestMultiprocessing(variousTreesMultiprocessinng, testRow)
        finalResult = random_forest.getFinalResult(classificationResult)
        if finalResult == testRow[columnToTestIndex]:
            rightAnswersCount += 1
    print "evaluated correctly {} out of {} tests".format(rightAnswersCount, len(testSet))
    print

# Test random forest with max elements in node
# Params = columnToTestIndex = 11, numberOfTrees = 3, minNodes = 15
maxElementsInNodeTest(11, 5, 4)
maxElementsInNodeTest(11, 3, 4)

# Test random forest with max Heigth
# Params = columnToTestIndex = 11, numberOfTrees = 3, maxHeigth = 15
maxHeigthTest(11, 5, 15)
maxHeigthTest(11, 3, 15)
