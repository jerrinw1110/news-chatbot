from openai import OpenAI
import os
from dotenv import load_dotenv
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from wordcloud import WordCloud





from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def format_message(role, content):
    return {"role": role, "content": content}


def get_response_from_model_1(messages):
    completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages,
    )
    content = completion.choices[0].message.content
    return content

def get_response_from_model_2(messages):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=messages,
    )
    content = completion.choices[0].message.content
    return content


def bias(quotes):

    instructions = f"""
    Here is a summary of a news article, try to classify the article based on media bias into one of 5 discrete classess: "far-left", "left", "center", "right", "far-right". 
    Start this classification with the tag *CLASSIFICATION:*
    Give an explanation for this classification in under 100 words. Please use bullet-points for this explanation.

    Quotes: {quotes}
    """

    message = format_message("system", instructions) # system means high priority 
    messages = [message] # ChatGPT API expects any message to be in a list
    response = get_response_from_model_1(messages)
    return response

# Preprocess texts for IC
def preprocess(text):
    # Basic text cleaning: convert to lowercase, remove punctuation, etc.
    import re
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text



load_dotenv()

key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key = key)




topic = "politics"
n=10

#Path to our data
directoryToSummarize = join("BBC News Summary","News Articles", topic)
#Path to our 'label'
directoryOfSummaries = join("BBC News Summary", "Summaries", topic)

directoryOfAISummaries = join("GeneratedSummaries","Summaries")


#A list of the files in the data directory
onlyfiles = [f for f in listdir(directoryToSummarize) if isfile(join(directoryToSummarize, f))]

#A list of the files in the 'label' directory
onlySummaries = [f for f in listdir(directoryOfSummaries) if isfile(join(directoryOfSummaries, f))]


onlyAISummaries = [f for f in listdir(directoryOfAISummaries) if isfile(join(directoryOfAISummaries, f))]

onlyfiles = sorted(onlyfiles, key=os.path.basename)

onlySummaries = sorted(onlySummaries, key=os.path.basename)

onlyAISummaries = sorted(onlyAISummaries, key=os.path.basename)


#Stores the data content in a list
contentList = []
BiasOfSummaries = []

#Read in our content
for filePath in onlyAISummaries:
    with open(join(directoryOfAISummaries,filePath), 'r') as file:
        contentList.append(file.read())

for i in range(n):
    BiasOfSummaries.append(bias(contentList[i]))


for i,suma in enumerate(BiasOfSummaries):
    with open("GeneratedSummaries" + os.sep + "BiasOfSummaries" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma)
        file.close()
