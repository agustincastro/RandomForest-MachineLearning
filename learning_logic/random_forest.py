import multiprocessing
import time
from learning_logic import decision_tree


##  --------------------   FOREST CREATION --------------------------
##  =================================================================

# Creates a decision tree for each subset inside dataset. The structure of the tree is determined by
# the buildTreeFunction parameter
def createDecisionTrees(dataSets, buildTreeFunction = decision_tree.buildTree, **kwargs):
    decisionTrees = []
    start_time = time.time()
    for subset in dataSets:
        decisionTrees.append(buildTreeFunction(subset, **kwargs))
    print("--- {} seconds to create trees sequential ---".format(time.time() - start_time))
    return decisionTrees


# Creates a decision tree for each subset inside dataset using paralel programming.
# The structure of the tree is determined by the buildTreeFunction parameter.
def createDecisionTreesMultiprocessing(dataSets, buildTreeFunction = decision_tree.buildTree, **kwargs):
    # Define an output queue
    output = multiprocessing.Queue()
    # Defines process function
    def seedTree(subset, output, **kwargs):
        output.put(buildTreeFunction(subset, **kwargs))
    # Setup a list of processes that we want to run
    processes = [multiprocessing.Process(target=seedTree, args=(dataSets[x], output), kwargs=kwargs) for x in range(0, len(dataSets))]
    # Run processes
    start_time = time.time()
    for p in processes: p.start()
    # Exit the completed processes
    for p in processes: p.join()
    print("--- {} seconds to create trees concurrently(multiprocessing)---".format(time.time() - start_time))
    # Get process results from the output queue
    return [output.get() for p in processes]


# Creates a decision tree for each subset inside dataset using a pool of processes.
# The structure of the tree is determined by the buildTreeFunction parameter.
def createDecisionTreesPool(dataSets, buildTreeFunction = decision_tree.buildTree, **kwargs):
    processes = 4
    # Sets max number of concurrent processes in pool if the value is sent in kwargs
    if 'processes' in kwargs:
        processes = kwargs.pop('processes')
    pool = multiprocessing.Pool(processes=processes)
    start_time = time.time()
    # Runs processes, the buildtree function is executed asynchronically until all the results are retrieved with get()
    results = [pool.apply_async(buildTreeFunction, (dataSets[x], ), kwargs) for x in range(0, len(dataSets))]
    output = [r.get() for r in results]
    print("--- {} seconds to create trees concurrently(pool)---".format(time.time() - start_time))
    return output



##  --------------------   FOREST CLASSIFICATION  --------------------------
##  ========================================================================

# Classifies a random forest and returns a list of tentative classifications for a test
def classifyForest(decisionTrees, test):
    results = [decision_tree.classifyInTree(tree, test) for tree in decisionTrees]
    return results


# Classifies a random forest and returns a list of tentative classifications for a test
# The classification against the forests are made in paralel using multiprocesses
def classifyForestMultiprocessing(decisionTrees, test):
    # Define an output queue
    output = multiprocessing.Queue()
    # Defines process function
    def classify(tree, test, output):
        output.put(decision_tree.classifyInTree(tree, test))
    # Setup a list of processes that we want to run
    processes = [multiprocessing.Process(target=classify, args=(tree, test, output)) for tree in decisionTrees]
    # Run processes
    start_time = time.time()
    for p in processes: p.start()
    # Exit the completed processes
    for p in processes: p.join()
    #print("--- %s seconds to classify trees concurrently(multiprocessing)---" % (time.time() - start_time))
    # Get process results from the output queue
    return [output.get() for p in processes]


# Returns the most repeated element in the result set provided by the classification of the random forest
def getFinalResult(classifications):
    return max(set(classifications), key=classifications.count)
















