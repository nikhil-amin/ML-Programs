# (QUESTION: 05)
# Write a program to implement the naïve Bayesian classifier for a sample training
# data set stored as a .CSV file. Compute the accuracy of the classifier, considering few
# test data sets.

import csv
import random
import math

def loadCsv(filename):
	lines=csv.reader(open(filename,"r"));
	dataset=list(lines)
	for i in range(len(dataset)):
		dataset[i]=[float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset,splitRatio):
	trainsize=int(len(dataset)*splitRatio);
	trainset=[]
	copy=list(dataset);
	while(len(trainset))<trainsize:
		index=random.randrange(len(copy));
		trainset.append(copy.pop(index))
	return[trainset,copy]

def seperateByClass(dataset):
	seperated={}
	for i in range(len(dataset)):
		vector=dataset[i]
		if(vector[-1] not in seperated):
			seperated[vector[-1]]=[]
		seperated[vector[-1]].append(vector)
	return seperated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg=mean(numbers)
	variance=sum([pow(x-avg,2)for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries=[(mean(attribute),stdev(attribute))for attribute in zip(*dataset)];
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	seperated=seperateByClass(dataset);
	summaries={}
	for classvalue,instances in seperated.items():
		summaries[classvalue]=summarize(instances)
	return summaries
	
def calculateprobability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return(1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateclassprobabilities(summaries,inputvector):
	probabilities={}
	for classvalue,classsummaries in summaries.items():
		probabilities[classvalue]=1
	for i in range(len(classsummaries)):
		mean,stdev=classsummaries[i]
		x=inputvector[i]
		probabilities[classvalue]*=calculateprobability(x,mean,stdev)
	return probabilities
		
	
def predict(summaries,inputvector):
	probabilities=calculateclassprobabilities(summaries,inputvector)
	bestlabel,bestpro=None,-1
	
	for classvalue,probability in probabilities.items():
		if bestlabel is None or probability>bestprob:
			bestprob=probability
			bestlabel=classvalue
	return bestlabel


def getPredictions(summaries,testset):
	predictions=[]
	for i in range(len(testset)):
		result=predict(summaries,testset[i])
		predictions.append(result)
	return predictions
	
def getAccuracy(testset,predictions):
	correct=0
	for i in range(len(testset)):
		if testset[i][-1]==predictions[i]:
			correct+=1
	return(correct/float(len(testset)))*100
	
def main():
	filename='PRG_5.csv'
	splitRatio=0.67
	dataset=loadCsv(filename);
	trainingset,testset=splitDataset(dataset,splitRatio)
	print('split{0} rows into train={1}and test={2} rows'.format(len(dataset),len(trainingset),len(testset)))
	
	summaries=summarizeByClass(trainingset);
	predictions=getPredictions(summaries,testset)
	accuracy=getAccuracy(testset,predictions)
	print('accuracy of the classifier is:{0}%'.format(accuracy))

main()	
