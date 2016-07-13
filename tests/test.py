from utils import dataset as datasetModule
from learning_logic import decision_tree, random_forest
import pkg_resources, os
import csv


def test1(rowToTestIndex = 11, numberOfTrees = 3):
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
    testSet = datasetModule.getTestSet(dataset,rowToTestIndex)
    # Delete balanced testSet from original dataset
    dataset = [x for x in dataset if x not in testSet]
    testRow = testSet[0]

    #Split dataSet into different subsets in order to create decision trees
    subsets = datasetModule.randomSplit(dataset, numberOfTrees)

    # Creates random Forest
    print('***** Creating random forest with ' + str(numberOfTrees) +' trees *****')
    variousTreesMultiprocessinng = random_forest.createDecisionTreesMultiprocessing(subsets, decision_tree.buildTreeWithMaxElementsInNode, minNodes = 1)
    #variousTreesMultiprocessinng = random_forest.createDecisionTreesMultiprocessing(subsets, decision_tree.buildTreeWithHeigth, maxHeigth = 15)

    for i in variousTreesMultiprocessinng:
        decision_tree.printtree(i)

    # Classify testSet against the forest
    rightAnswersCount = 0
    for testRow in testSet:
        classificationResult = random_forest.classifyForestMultiprocessing(variousTreesMultiprocessinng, testRow)
        finalResult = random_forest.getFinalResult(classificationResult)
        if finalResult == testRow[rowToTestIndex]:
            rightAnswersCount += 1
    print "evaluated correctly {} out of {} tests".format(rightAnswersCount, len(testSet))



test1(11, 5)
