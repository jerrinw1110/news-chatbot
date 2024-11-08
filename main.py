from openai import OpenAI
from os import listdir
from os.path import isfile, join
import os
import summaries
import Bert



#TODO: Make this recursively find all data 

topic = "politics"

#Path to our data
directoryToSummarize = join("BBC News Summary","News Articles", topic)
#Path to our 'label'
directoryOfSummaries = join("BBC News Summary", "Summaries", topic)

#A list of the files in the data directory
onlyfiles = [f for f in listdir(directoryToSummarize) if isfile(join(directoryToSummarize, f))]

#A list of the files in the 'label' directory
onlySummaries = [f for f in listdir(directoryOfSummaries) if isfile(join(directoryOfSummaries, f))]

#Stores the data content in a list
contentList = []

#Store the summary content from labels
summaryList = []

#Read in our content
for filePath in onlyfiles:
    with open(join(directoryToSummarize,filePath), 'r') as file:
        contentList.append(file.read())
#Read in our summaries
for filePath in onlySummaries:
    with open(join(directoryOfSummaries,filePath), 'r') as file:
        summaryList.append(file.read())


#Predicted Summaries List
AIsummaries = []

#Predicted Bias List
Bias = []

PredictedBiasThenSummaries = []

#Call ChatGPT to summarize
for i in range(3):
    AIsummaries.append(summaries.summarize((contentList[i]), len(summaryList[i].split(" "))))

    Bias.append(summaries.bias(contentList[i]))

    PredictedBiasThenSummaries.append(summaries.summarizeANDBias(contentList[i], len(summaryList[i].split(" "))))

#Write to files

for i,suma in enumerate(AIsummaries):
    with open("GeneratedSummaries" + os.sep + "Summaries" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma)
        file.close()

for i,suma in enumerate(Bias):
    with open("GeneratedSummaries" + os.sep + "Detected Bias" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma)
        file.close()

for i,suma in enumerate(PredictedBiasThenSummaries):
    with open("GeneratedSummaries" + os.sep + "BiasThenSummary" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma)
        file.close()



for i,suma in enumerate(AIsummaries):
    with open("GeneratedSummaries" + os.sep + "BertscoreForSummaries" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        
        scoredict = Bert.evaluate_with_bertscore(suma, summaryList[i])
        
        file.write(str(scoredict))
        file.close()


for i,suma in enumerate(PredictedBiasThenSummaries):
    with open("GeneratedSummaries" + os.sep + "BertScoreForBiasThenSummary" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        
        scoredict = Bert.evaluate_with_bertscore(suma, summaryList[i])
        
        file.write(str(scoredict))
        file.close()

"""
TODO:
-Compare human summaries to ChatGPT summaries
-Automate and save the three tasks: Summarize, Get Bias, and Get Bias and then summarize

-Possible Comparison methods: BERT, rouge, some from class

-Set up scripting to to seperate comparison and summary generation
-Set up arguments for number of summaries to generate and whether to calculate similarity


"""