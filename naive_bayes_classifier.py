import csv
import random
import math
import numpy
from datetime import datetime

def loadCsv(filename):
	lines = csv.reader(open('data/gold-price.csv', "rb"))
	next(lines) # skip header
	gold = list(lines)
	goldResult = {}
	for i in range(len(gold)):
		goldResult[gold[i][0]] = float(gold[i][1])
	
	lines = csv.reader(open('data/oil-price.csv', "rb"))
	next(lines) # skip header
	oil = list(lines)
	oilResult = {}
	for i in range(len(oil)):
		oilResult[oil[i][0]] = float(oil[i][1])
		
	lines = csv.reader(open(filename, "rb"))
	next(lines) # skip header
	dataset = list(lines)
	indicesOfOpenClose = [6,2]
	result = []
	for i in range(len(dataset)-3):
		tupOpenCloseOfDay0 = [float(dataset[i][j]) for j in indicesOfOpenClose]
		deltaOpenCloseOfDay0 = tupOpenCloseOfDay0[1] - tupOpenCloseOfDay0[0]
		labelUpDownOfDay0 = numpy.sign(deltaOpenCloseOfDay0)
		tupOpenCloseOfDay1 = [float(dataset[i+1][j]) for j in indicesOfOpenClose]
		deltaOpenCloseOfDay1 = tupOpenCloseOfDay1[1] - tupOpenCloseOfDay1[0]
		if deltaOpenCloseOfDay1 >= 0:
			labelUpDownOfDay1 = 1.0
		else:
			labelUpDownOfDay1 = -1.0
		tupOpenCloseOfDay2 = [float(dataset[i+2][j]) for j in indicesOfOpenClose]
		deltaOpenCloseOfDay2 = tupOpenCloseOfDay2[1] - tupOpenCloseOfDay2[0]
		if deltaOpenCloseOfDay2 >= 0:
			labelUpDownOfDay2 = 1.0
		else:
			labelUpDownOfDay2 = -1.0
		tupOpenCloseOfDay3 = [float(dataset[i+3][j]) for j in indicesOfOpenClose]
		deltaOpenCloseOfDay3 = tupOpenCloseOfDay3[1] - tupOpenCloseOfDay3[0]
		if deltaOpenCloseOfDay3 >= 0:
			labelUpDownOfDay3 = 1.0
		else:
			labelUpDownOfDay3 = -1.0
		deltaCloseOfDay2Day0 = tupOpenCloseOfDay2[1] - tupOpenCloseOfDay0[1];
		deltaCloseOfDay2Day1 = tupOpenCloseOfDay2[1] - tupOpenCloseOfDay1[1];
		weekday = datetime.strptime(dataset[i+3][3], "%Y-%m-%d").weekday()
		if goldResult.get(dataset[i+3][3]) != None:
			goldPrice = goldResult.get(dataset[i+3][3])
		if oilResult.get(dataset[i+3][3]) != None:
			oilPrice = oilResult.get(dataset[i+3][3])
		result.append([deltaCloseOfDay2Day0, deltaCloseOfDay2Day1, weekday, goldPrice, oilPrice, labelUpDownOfDay3])
	return result

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = 0
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			if isinstance(x, basestring):
				x = float(x)
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	filename = 'data/hsi.csv'
	splitRatio = 0.8
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()