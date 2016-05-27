import csv
import random
import math

#Loads file in CSV format and processes its data
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


# Separates the training dataset instances by class value so that we can calculate statistics for each class
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated


# For a given list of instances (for a class value) we can calculate the mean and the standard deviation for each attribute.
# The zip function groups the values for each attribute across our data instances into their own lists so that we
# can compute the mean and standard deviation values for the attribute.
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries


# The standard deviation describes the variation of spread of the data, and we will use it to characterize the
# expected spread of each attribute in our Gaussian distribution when calculating probabilities.
# The standard deviation is calculated as the square root of the variance. The variance is calculated as the
# average of the squared differences for each attribute value from the mean. Note we are using the N-1 method,
# which subtracts 1 from the number of attribute values when calculating the variance.
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)


# The mean is the central middle or central tendency of the data, and we will use it as the middle of
# our gaussian distribution when calculating probabilities.
def mean(numbers):
	return sum(numbers)/float(len(numbers))


# Separates our training dataset into instances grouped by class. Then calculateS the summaries for each attribute.
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
