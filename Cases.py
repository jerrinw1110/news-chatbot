#File these down a bit
import pandas as pd
import sklearn.feature_extraction
import summaries

#For Case 1
import Bert
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity





#Note, I want these running independantly

#Some C
POLITICS_OFFSET = 898
ARTICLE_COUNT = 400

#If diverse flag is set, pull from 2nd dataset, otherwise just BBC
def pull_articles(diverse_Flag):

    
    if not diverse_Flag:
        BBC_DataFrame = pd.read_csv("BBC_News_Summary.csv")
        return BBC_DataFrame[POLITICS_OFFSET:POLITICS_OFFSET+ARTICLE_COUNT]
    else: 
        Extra_Articles = pd.read_excel("articles_400.xlsx")
        #Shuffle our data
        Extra_Articles = Extra_Articles.sample(frac=1)
        return Extra_Articles[0:ARTICLE_COUNT]

# Preprocess texts for IC
def preprocess(text):
    # Basic text cleaning: convert to lowercase, remove punctuation, etc.
    import re
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

def draw_distributions(Articles_DataFrame: pd.DataFrame):
    
    #Set up IC vectors, pulling from dataframe
    IC_vector_for_humans = Articles_DataFrame["Information Coverage for Humans"].to_numpy()
    IC_vector_for_ai = Articles_DataFrame["Information Coverage for AI"].to_numpy()

    #Set up Bert vectors, from dataframe
    bert_precision = Articles_DataFrame["Bert Mean Precision"].to_numpy()
    bert_recall = Articles_DataFrame["Bert Mean Recall"].to_numpy()
    bert_f1 = Articles_DataFrame["Bert Mean F1"].to_numpy()

    #FIGURE 1
    plt.figure()

    x = np.linspace(0, 1, 200)

    y_ICHUMAN = gaussian(x, np.mean(IC_vector_for_humans), np.std(IC_vector_for_humans))
    y_ICAI = gaussian(x, np.mean(IC_vector_for_ai), np.std(IC_vector_for_ai))

    plt.plot(x, y_ICHUMAN, color = "blue")
    plt.plot(x, y_ICAI, color="red")
    plt.title("Normal Distribution of Information Coverage")
    plt.xlabel("Information Coverage")
    plt.ylabel("Probability of Information Coverage for a Summary")
    plt.savefig("Information Coverage Distributions.png")

    #FIGURE 2
    plt.figure()

    x = np.linspace(0, 1, 200)

    y_BertPrec = gaussian(x, np.mean(bert_precision), np.std(bert_precision))
    y_BertRec = gaussian(x, np.mean(bert_recall), np.std(bert_recall))
    y_BertF1 = gaussian(x, np.mean(bert_f1), np.std(bert_f1))


    plt.plot(x, y_BertPrec, color = "blue")
    plt.plot(x, y_BertRec, color="red")
    plt.plot(x, y_BertF1, color="green")
    plt.title("Normal Distribution for Bert Precision(blue), Recall(Red), F1(Green)")
    plt.xlabel("Performance")
    plt.ylabel("Probability of Performance for a Summary")
    plt.savefig("Bert Score Distributions.png")




    return

#For graphing method
def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

#For Claim 2, draw the bar chart
def draw_bar_chart(bias_classes, ai_count, human_count):


    # set width of bar 
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8)) 

 
    # Set position of bar on X axis 
    br1 = np.arange(5) 
    br2 = [x + barWidth for x in br1] 

    # Make the plot
    plt.bar(br1, ai_count, color ='r', width = barWidth, 
            edgecolor ='grey', label ='AI') 
    plt.bar(br2, human_count, color ='b', width = barWidth, 
            edgecolor ='grey', label ='HUMAN') 

    # Adding Xticks 
    plt.xlabel('Bias', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Number of Detections', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barWidth for r in range(5)], 
            bias_classes)

    plt.legend()
    plt.savefig("test.png")


    return




#Claim 1: ChatGPT 4.0 is able to summarize an news article equivalently to a human being and can provide equal or higher information coverage compared to human work.
#Note, this must run before Claim 3
"""
Whats Needed:
-Exclusively BBC articles
-100 articles
-100 ai summaries
-100 human summaries
-Going to need to store these in a file:
-Store ai summaries
-IC
-Calculate distribution and graph both
"""
"""
Functions:
pull_articles
draw_distributions

"""
def Case1():
    Articles_DataFrame = pull_articles(diverse_Flag=False)
    Articles_DataFrame.reset_index(inplace=True, drop=True)
  

    #Do ai summaries
    AI_Summaries = []
    for index, series in Articles_DataFrame.iterrows():
        article = series["Article"]
        AI_Summaries.append(summaries.summarize(article, len(article)))

    Articles_DataFrame["AI Summary"] = AI_Summaries


    #calculate bert score
    Bert_Scores = []
    for index, series in Articles_DataFrame.iterrows():
        Bert_Scores.append(Bert.evaluate_with_bertscore(series["Summary"], series["AI Summary"]))

    #Reformat into a way to store in csv
    bert_precision = []
    bert_recall = []
    bert_f1 = []
    for val in Bert_Scores:
        bert_precision.append(val["precision"])
        bert_recall.append(val["recall"])
        bert_f1.append(val["f1"])

    Articles_DataFrame["Bert Mean Precision"] = bert_precision
    Articles_DataFrame["Bert Mean Recall"] = bert_recall
    Articles_DataFrame["Bert Mean F1"] = bert_f1

    



    #calculate Information Coverage
    IC_of_human = []
    IC_of_AI = []
    vectorizer = TfidfVectorizer()
    Articles = Articles_DataFrame["Article"]
    for index, series in Articles_DataFrame.iterrows():
        #Preprocess article
        article = preprocess(series["Article"])
        aiSummary = preprocess(series["AI Summary"])
        humanSummary = preprocess(series["Summary"])

        tfIDFMatrix = vectorizer.fit_transform([article, aiSummary])

        similarity = cosine_similarity(tfIDFMatrix[0:1], tfIDFMatrix[1:2])

        IC_of_AI.append(similarity[0][0])


        #Calculate IC of human summary

        tfIDFMatrix = vectorizer.fit_transform([article, humanSummary])
        
        similarity = cosine_similarity(tfIDFMatrix[0:1], tfIDFMatrix[1:2])

        IC_of_human.append(similarity[0][0])

    Articles_DataFrame["Information Coverage for Humans"] = IC_of_human
    Articles_DataFrame["Information Coverage for AI"] = IC_of_AI

    #Make csv file
    #Save ai summaries, bert score, and IC
    Articles_DataFrame.to_csv("Case_1.csv")

    return
    



#Claim 2: ChatGPT 4.0 is able to correctly classify an news article into one of the five bias group and has a equivalent or higher accuracy compared to pre-trained media bias classifiers
"""
Whats Needed:
-Diverse articles 
-100 articles
-Bias detect via ChatGPT 4o-mini
-Bar graph via matplot lib ***** NEW DEPENDANCY
"""
def Case2():
    #Pull in our data, which is randomized
    Articles_df = pull_articles(diverse_Flag=True)
    titles = Articles_df["title"].to_list()
    body = Articles_df["body"].to_list()

    #Format title and body into one string.
    articles = []
    for i in range(len(titles)):
        articles.append("Article Title: " + titles[i] + "\nBody: " + body[i])

    #Ping Chat GPT for its bias classification
    raw_ai_bias = []
    for article in articles:
        raw_ai_bias.append(summaries.bias(article))

    #Now we want to create a bar graph
    articleClassificationByAI = []
    explanationOfClassification = []
    for bias in raw_ai_bias:
        biasList = bias.split("\n")
        articleClassificationByAI.append(biasList[0].strip())
        explanationOfClassification.append(biasList[1])

    articleClassification = Articles_df["bias"].to_list()
    bias_classes = ["Left", "Lean Left", "Center", "Lean Right", "Right"]

    human_classification_count = np.array([articleClassification.count(bias_classes[i]) for i in range(5)])
    ai_classification_count = np.array([articleClassificationByAI.count(bias_classes[i]) for i in range(5)])
    

    draw_bar_chart(bias_classes, ai_classification_count, human_classification_count)

    #Time to save csv
    #This might need to be updated to use the formatted data
    Articles_df["Ai bias and explanation"] = raw_ai_bias

    Articles_df.to_csv("Case_2.csv")

    return

#Claim 3: The bias results based on the ChatGPT summary will (not) be deviated from the results based on the original article.
"""
Whats Needed:
-Diverse Articles
-100 articles
-100 ai summaries
-Detect Bias in all 200
-Save only the Bias leaning
-Bar graph, seperating ai bias and article bias
"""
def Case3():


    return



#Main controls (for now?)
"""
Case1()
Case1_DataFrame = pd.read_csv("Case_1.csv")
draw_distributions(Case1_DataFrame)

Case2()

Case3()

"""