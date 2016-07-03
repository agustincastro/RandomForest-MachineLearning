import math

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
