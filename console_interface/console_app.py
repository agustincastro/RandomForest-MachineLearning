from utils import utils
from learning_logic import supervised


def main():
    filename = 'pima-indians-diabetes.data.csv'
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


main()
