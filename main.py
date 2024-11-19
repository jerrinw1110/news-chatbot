from openai import OpenAI
from os import listdir
from os.path import isfile, join
import os

import sklearn.feature_extraction
import summaries
import Bert

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocess texts for IC
def preprocess(text):
    # Basic text cleaning: convert to lowercase, remove punctuation, etc.
    import re
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text


"""
#Needed parameters
number of summaries to make
summary, bias, bias then summary, or all three
2-way comparison, three way comparison


"""
#TODO: Make this recursively find all data 

topic = "politics"
n=10

#Path to our data
directoryToSummarize = join("BBC News Summary","News Articles", topic)
#Path to our 'label'
directoryOfSummaries = join("BBC News Summary", "Summaries", topic)

#A list of the files in the data directory
onlyfiles = [f for f in listdir(directoryToSummarize) if isfile(join(directoryToSummarize, f))]

#A list of the files in the 'label' directory
onlySummaries = [f for f in listdir(directoryOfSummaries) if isfile(join(directoryOfSummaries, f))]

onlyfiles = sorted(onlyfiles, key=os.path.basename)

onlySummaries = sorted(onlySummaries, key=os.path.basename)



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


#Call ChatGPT to summarize
for i in range(n):
    AIsummaries.append(summaries.summarize((contentList[i]), len(summaryList[i].split(" "))))

    Bias.append(summaries.bias(contentList[i]))



#Write to files

for i,suma in enumerate(AIsummaries):
    with open("GeneratedSummaries" + os.sep + "Summaries" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma)
        file.close()

for i,suma in enumerate(Bias):
    with open("GeneratedSummaries" + os.sep + "Detected Bias" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma)
        file.close()





for i,suma in enumerate(AIsummaries):
    with open("GeneratedSummaries" + os.sep + "BertscoreForSummaries" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        
        scoredict = Bert.evaluate_with_bertscore(suma, summaryList[i])
        
        file.write(str(scoredict))
        file.close()


for i,suma in enumerate(AIsummaries):
    vectorizer = TfidfVectorizer()
    
    #Calculate IC of gen summary
    article = preprocess(contentList[i])
    aiSummary = preprocess(suma)

    tfIDFMatrix = vectorizer.fit_transform([article, aiSummary])

    similarity = cosine_similarity(tfIDFMatrix[0:1], tfIDFMatrix[1:2])

    coverage_score = similarity[0][0]

    print("The IC of the ai summary is: " + str(coverage_score))

    #Calculate IC of human summary
    humanSummary = preprocess(summaryList[i])
    tfIDFMatrix = vectorizer.fit_transform([article, humanSummary])
    
    similarity = cosine_similarity(tfIDFMatrix[0:1], tfIDFMatrix[1:2])

    coverage_score = similarity[0][0]

    
    print("The IC of the human summary is: " + str(coverage_score))






"""
TODO:
-Compare human summaries to ChatGPT summaries
-Automate and save the three tasks: Summarize, Get Bias, and Get Bias and then summarize

-Possible Comparison methods: BERT, rouge, some from class

-Set up scripting to to seperate comparison and summary generation
-Set up arguments for number of summaries to generate and whether to calculate similarity


"""