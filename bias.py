from openai import OpenAI
import os
from dotenv import load_dotenv
from os import listdir
from os.path import isfile, join


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
    Here is a news article, try to classify the article based on media bias into one of 5 discrete classess: "far-left", "left", "center", "right", "far-right". 
    Start this classification with the tag *CLASSIFICATION:*
    Give an explanation for this classification in under 100 words. Please use bullet-points for this explanation.

    Quotes: {quotes}
    """

    message = format_message("system", instructions) # system means high priority 
    messages = [message] # ChatGPT API expects any message to be in a list
    response = get_response_from_model_1(messages)
    response2 = get_response_from_model_2(messages)
    return response, response2

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

#A list of the files in the data directory
onlyfiles = [f for f in listdir(directoryToSummarize) if isfile(join(directoryToSummarize, f))]

#A list of the files in the 'label' directory
onlySummaries = [f for f in listdir(directoryOfSummaries) if isfile(join(directoryOfSummaries, f))]

onlyfiles = sorted(onlyfiles, key=os.path.basename)

onlySummaries = sorted(onlySummaries, key=os.path.basename)



#Stores the data content in a list
contentList = []
Bias = []


#Read in our content
for filePath in onlyfiles:
    with open(join(directoryToSummarize,filePath), 'r') as file:
        contentList.append(file.read())

for i in range(n):
    response1,response2 = bias(contentList[i])
    Bias.append([response1,response2])

for i,suma in enumerate(Bias):
    with open("GeneratedSummaries" + os.sep + "BiasFromModel1" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma[0])
        file.close()
    with open("GeneratedSummaries" + os.sep + "BiasFromModel2" + os.sep  + topic + "{:0>3}.txt".format(i), 'w') as file:
        file.write(suma[1])
        file.close()

for b in Bias:
    bias1 = b[0]
    bias2 = b[1]

    bias1 = "\n".join(bias1.split("\n")[1:])
    bias2 = "\n".join(bias2.split("\n")[1:])
    

    vectorizer = TfidfVectorizer()
    
    #Calculate IC of gen summary
    bias1 = preprocess(bias1)
    bias2 = preprocess(bias2)

    tfIDFMatrix = vectorizer.fit_transform([bias1, bias2])

    similarity = cosine_similarity(tfIDFMatrix[0:1], tfIDFMatrix[1:2])

    coverage_score = similarity[0][0]

    print("The similarity of the two bias scores is: " + str(coverage_score))



