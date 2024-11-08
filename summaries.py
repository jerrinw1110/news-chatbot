from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key = key)
def format_message(role, content):
    return {"role": role, "content": content}


def get_response(messages):
    completion = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages,
    )
    content = completion.choices[0].message.content
    return content

def summarize(quotes, length):

    instructions = f"""
    Here is a news article, summarize the article in less than {length} words while still maintaining information.

    Quotes: {quotes}
    """

    message = format_message("system", instructions) # system means high priority 
    messages = [message] # ChatGPT API expects any message to be in a list
    response = get_response(messages)
    return response

def bias(quotes):

    instructions = f"""
    Here is a news article, try to classify the article based on media bias into one of 5 discrete classess: "far-left", "left", "center", "right", "far-right". 
    Give an explanation for this classification in under 100 words. Try to use bullet-points.

    Quotes: {quotes}
    """

    message = format_message("system", instructions) # system means high priority 
    messages = [message] # ChatGPT API expects any message to be in a list
    response = get_response(messages)
    return response

def summarizeANDBias(quotes, length):

    instructions = f"""
    Here is a news article, first try to classify the article based on media bias into one of 5 discrete classess: "far-left", "left", "center", "right", "far-right". Give an explanation for this classification.
    Then summarize the article in less than {length} words while still maintaining information.
    
    Quotes: {quotes}
    """

    message = format_message("system", instructions) # system means high priority 
    messages = [message] # ChatGPT API expects any message to be in a list
    response = get_response(messages)
    return response
