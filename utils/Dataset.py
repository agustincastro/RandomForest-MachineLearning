import collections
import csv
import random

# Example of data structure
my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]


# Loads file in CSV format and processes its data
def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):  # Convert all data from string to int
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# split the data into a training dataset to make predictions and a test dataset
# that we can use to evaluate the accuracy of the model. Splits the data set randomly into train
# and datasets with a ratio of for example 67% train and 33% test (this is a common ratio for testing
# an algorithm on a dataset)
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


# Removes every row that has blank data in order not to bias the algorithm
def normalizeDataset(dataSet):
    rowLength = len(dataSet[0])
    dataSetLength = len(dataSet)
    lineIndex = 0
    for lineNumber in range(0, dataSetLength):
        line = dataSet[lineIndex]
        deletedLine = False
        for col in range(0, rowLength):
            if (line[col] == ''):
                del dataSet[lineIndex]
                deletedLine = True
            if deletedLine: break
        if not deletedLine: lineIndex += 1
    return dataSet


# Removes entire column from the dataset by index, columnIndex starts at 0
def removeColumn(dataSet, columnIndex):
    for row in dataSet:
        del row[columnIndex]


# Moves one column in the dataset to the last of the dataset
def postponeColumn(rows, columToPostpone):
    columToPostpone = columToPostpone - 1
    for row in rows:
        row.append(row[columToPostpone])
        del row[columToPostpone]


def printDataSet(dataSet):
    print("DATASET:")
    print('')
    for line in dataSet:
        print(line)


# Splits a dataset randomly into a number of datasets
def randomSplit(dataSet, subsetQuantity):
    startingIndex = 0
    dataSets = []
    subsetLength = int(round(len(dataSet) / subsetQuantity))
    random.shuffle(dataSet)
    for i in range(0, subsetQuantity):
        dataSets.append(dataSet[startingIndex: startingIndex + subsetLength])
        startingIndex += subsetLength
    return dataSets


def getTestSet(dataSet, column, divisionPercentage=30):
    columnList = [row[column] for row in dataSet]
    ocurrences = collections.Counter(columnList)
    percentages = {}

    # The percentages dictionary contains the quantites of each attribute class we need to take
    # in order to get a well balanced test set
    for var in ocurrences:
        percentages[var] = (ocurrences[var] * divisionPercentage) / 100
    # We shuffle randomly the dataset in order to get always a different testset
    print(percentages)
    random.shuffle(dataSet)
    testSet = []
    for row in dataSet:
        value = row[column]
        if (percentages[value] > 0):
            testSet.append(row)
            percentages[value] -= 1
    return testSet
