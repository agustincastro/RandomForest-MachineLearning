from utils import utils
from learning_logic import supervised

def main():
	filename = 'pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataset = utils.loadCsv(filename)
	trainingSet, testSet = supervised.splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = supervised.summarizeByClass(trainingSet)
	# test model
	predictions = supervised.getPredictions(summaries, testSet)
	accuracy = supervised.getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()