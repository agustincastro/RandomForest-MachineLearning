from utils import utils
from learning_logic import supervised, decision_tree
import pkg_resources, os
import csv



def decisionTreeMain():
    resource_package = 'resources'
    filename = 'titanic.train.csv'
    resource_path = os.path.join('training_data', filename)
    filePath = pkg_resources.resource_filename(resource_package, resource_path) # Gets path of file from another package

    lines = csv.reader(open(filePath, "rb"))
    dataset = list(lines)

    datasetEntropy = decision_tree.entropy(dataset)
    print('Entropy in {0} dataset: {1}').format(filename, str(datasetEntropy))



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
