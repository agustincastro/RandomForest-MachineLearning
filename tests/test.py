import csv

import os
import pkg_resources

from learning_logic import decision_tree, random_forest
from utils import dataset as datasetModule


def maxElementsInNodeTest(columnToTestIndex=11, numberOfTrees=3, minNodes=15):
    # ----- Import csv file -----
    resource_package = 'resources'
    filename = 'wine-color.csv'
    resource_path = os.path.join('training_data', filename)
    filePath = pkg_resources.resource_filename(resource_package,
                                               resource_path)  # Gets path of file from another package
    lines = csv.reader(open(filePath, "rb"))
    dataset = list(lines)

    #  ---- Prepare dataset for analysing  ------
    del dataset[0]  # removes headers from dataset
    datasetModule.normalizeDataset(dataset)

    # ----- Separates testSet from Dataset ------
    testSet = datasetModule.getTestSet(dataset, columnToTestIndex, 10)
    # Delete balanced testSet from original dataset
    dataset = [x for x in dataset if x not in testSet]

    # Split dataSet into different subsets in order to create decision trees
    subsets = datasetModule.randomSplit(dataset, numberOfTrees)

    # Creates random Forest
    print("***** Creating random forest with {} min nodes({}) trees *****".format(numberOfTrees, minNodes))
    variousTreesMultiprocessinng = random_forest.createDecisionTreesMultiprocessing(subsets,
                                                                                    decision_tree.buildTreeWithMaxElementsInNode,
                                                                                    minNodes=minNodes)

    # Classify testSet against the forest
    rightAnswersCount = 0
    for testRow in testSet:
        classificationResult = random_forest.classifyForestMultiprocessing(variousTreesMultiprocessinng, testRow)
        finalResult = random_forest.getFinalResult(classificationResult)
        if finalResult == testRow[columnToTestIndex]:
            rightAnswersCount += 1
    print("evaluated correctly {} out of {} tests".format(rightAnswersCount, len(testSet)))
    print()


def maxHeigthTest(columnToTestIndex=11, numberOfTrees=3, maxHeigth=15):
    # ----- Import csv file -----
    resource_package = 'resources'
    filename = 'wine-quality-red.csv'
    resource_path = os.path.join('training_data', filename)
    filePath = pkg_resources.resource_filename(resource_package,
                                               resource_path)  # Gets path of file from another package
    lines = csv.reader(open(filePath, "rb"))
    dataset = list(lines)

    #  ---- Prepare dataset for analysing  ------
    del dataset[0]  # removes headers from dataset
    dataset = datasetModule.normalizeDataset(dataset)

    # ----- Separates testSet from Dataset ------
    testSet = datasetModule.getTestSet(dataset, columnToTestIndex, 30)
    # Delete balanced testSet from original dataset
    dataset = [x for x in dataset if x not in testSet]

    # Split dataSet into different subsets in order to create decision trees
    subsets = datasetModule.randomSplit(dataset, numberOfTrees)

    # Creates random Forest
    print("***** Creating random forest with {} max heigth({}) trees. *****".format(numberOfTrees, maxHeigth))
    variousTreesMultiprocessinng = random_forest.createDecisionTrees(subsets, decision_tree.buildTreeWithHeigth,
                                                                     maxHeigth=15)

    # Classify testSet against the forest
    rightAnswersCount = 0
    for testRow in testSet:
        classificationResult = random_forest.classifyForestMultiprocessing(variousTreesMultiprocessinng, testRow)
        finalResult = random_forest.getFinalResult(classificationResult)
        if finalResult == testRow[columnToTestIndex]:
            rightAnswersCount += 1
    print("evaluated correctly {} out of {} tests".format(rightAnswersCount, len(testSet)))
    print()


# Test random forest with max elements in node
# Params = columnToTestIndex = 11, numberOfTrees = 3, minNodes = 15
# maxElementsInNodeTest(12, 2, 200)
# maxElementsInNodeTest(12, 2, 50)
# maxElementsInNodeTest(12, 2, 20)
# maxElementsInNodeTest(12, 2, 5)


# Test random forest with max Heigth
# Params = columnToTestIndex = 11, numberOfTrees = 3, maxHeigth = 15
# maxHeigthTest(11, 1, 5)
maxHeigthTest(11, 3, 10)
# maxHeigthTest(11, 1, 15)
# maxHeigthTest(11, 1, 20)
