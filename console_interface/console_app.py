from utils import dataset as datasetModule
from learning_logic import decision_tree
import multiprocessing
import pkg_resources, os
import csv
import time



# Creates a decision tree for each subset inside dataset. The structure of the tree is determined by
# the buildTreeFunction parameter
def createDecisionTrees(dataSets, buildTreeFunction = decision_tree.buildTree):
    decisionTrees = []
    for subset in dataSets:
        decisionTrees.append(buildTreeFunction(subset))
    return decisionTrees



# Creates a decision tree for each subset inside dataset using paralel programming.
# The structure of the tree is determined by the buildTreeFunction parameter.
def createDecisionTreesMultiprocessing(dataSets, buildTreeFunction = decision_tree.buildTree):
    # Define an output queue
    output = multiprocessing.Queue()
    # Defines process function
    def seedTree(subset, output):
        output.put(buildTreeFunction(subset))
    # Setup a list of processes that we want to run
    processes = [multiprocessing.Process(target=seedTree, args=(dataSets[x], output)) for x in range(0, len(dataSets)-1)]
    # Run processes
    start_time = time.time()
    for p in processes: p.start()
    # Exit the completed processes
    for p in processes: p.join()
    print("--- %s seconds ---" % (time.time() - start_time))
    # Get process results from the output queue
    return [output.get() for p in processes]



def decisionTreeMain():
    # ----- Import csv file -----
    resource_package = 'resources'
    filename = 'titanic.train.csv'
    resource_path = os.path.join('training_data', filename)
    filePath = pkg_resources.resource_filename(resource_package, resource_path) # Gets path of file from another package
    lines = csv.reader(open(filePath, "rb"))
    dataset = list(lines)

    #  ---- Prepare dataset for analysing  ------
    del dataset[0] # removes headers from dataset
    dataset = datasetModule.normalizeDataset(dataset)
    #removeColumn(dataset, 3)

    print('***** Splits random subsets in order to create a random forest ****')
    subsets = datasetModule.randomSplit(dataset, 3)
    datasetModule.printDataSet(subsets)

    #printDataSet(dataset)

    #datasetEntropy = decision_tree.entropy(my_data) # 2.40
    #print('Entropy in {0} dataset: {1}').format(filename, str(datasetEntropy))
    datasetModule.postponeColumn(dataset, 2) # Shifts 'Survive' column to the last
    rowToClassify = dataset[1]
    del dataset[1]

    tree = decision_tree.buildTreeWithHeigth(dataset, maxHeigth=5)
    decision_tree.printtree(tree)
    print "Decidimos la siguiente fila:"
    print rowToClassify
    print "Resultado -> " + str(decision_tree.classifyInTree(tree, rowToClassify))




decisionTreeMain()
