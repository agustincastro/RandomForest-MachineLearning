from utils import dataset as datasetModule
from learning_logic import decision_tree, random_forest
import pkg_resources, os
import csv


def decisionTreeMain():
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
    #datasetModule.postponeColumn(dataset, 2) # Shifts 'Survive' column to the last
    #removeColumn(dataset, 3)

    print('***** Splits random subsets in order to create a random forest *****')
    subsets = datasetModule.randomSplit(dataset, 3)
    datasetModule.printDataSet(subsets)

    #printDataSet(dataset)

    #datasetEntropy = decision_tree.entropy(my_data) # 2.40
    #print('Entropy in {0} dataset: {1}').format(filename, str(datasetEntropy))

    testRow = dataset[1477]
    del dataset[1]

    #variousTrees = random_forest.createDecisionTrees(subsets, decision_tree.buildTreeWithMaxElementsInNode, minNodes = 100)
    #for i in variousTrees:
    #    decision_tree.printtree(i)

    #variousTreesPool = random_forest.createDecisionTreesPool(subsets, decision_tree.buildTreeWithMaxElementsInNode, processes = 5 ,minNodes = 100)
    #for i in variousTreesPool:
    #    decision_tree.printtree(i)

    variousTreesMultiprocessinng = random_forest.createDecisionTreesMultiprocessing(subsets, decision_tree.buildTreeWithMaxElementsInNode, minNodes = 100)
    for i in variousTreesMultiprocessinng:
        decision_tree.printtree(i)

    classificationResult = random_forest.classifyForestMultiprocessing(variousTreesMultiprocessinng, testRow)
    print classificationResult
    print "La clasificacion final es: " + random_forest.getFinalResult(classificationResult)
    #tree = decision_tree.buildTreeWithHeigth(dataset, maxHeigth=5)
    #decision_tree.printtree(tree)
    print "Decidimos la siguiente fila:"
    print testRow
   # print "Resultado -> " + str(decision_tree.classifyInTree(tree, testRow))




decisionTreeMain()
