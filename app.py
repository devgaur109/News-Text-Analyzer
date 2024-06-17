import concurrent.futures
import requests
import json
from exa_py import Exa
from newspaper import Article
from newspaper import Config
import re
from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import nltk
import string
from datetime import date
from nltk.corpus import stopwords

app = Flask(__name__)
today = date.today().strftime('%Y-%m-%d')

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10
# Define API keys
EXA_API_KEY = "4a1fea03-db83-41da-ae96-1d77ab50a26c"
BEARER_TOKEN = "b465a46a-ee54-4ef9-97e5-ca4c3cf753c2"

# Initialize Exa
exa = Exa(EXA_API_KEY)

# Helper function to make API calls to the LLM
def make_llm_api_call(messages):
    url = "https://api.awanllm.com/v1/chat/completions"

    payload = json.dumps({
        "model": "Awanllm-Llama-3-8B-Dolfin",
        "messages": messages,
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 512,
        "stream": False
    })

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {BEARER_TOKEN}"
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    content = ""
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
    return content

# Function to get search query from the news article
def get_search_query(news_article):
    messages = [
        {"role": "user", "content": "Extract all the main information from the news (headline). Summarize the news in the format that i can successfully search it on the internet. You can take help of any dates mentioned in the news to get a more refined question. Give it in the form of a question. Generate a single query only containing all the info. Don't generate or ouput any other text message except the query." + news_article}
    ]
    return make_llm_api_call(messages)

# Function to search the web using Exa API
def search_web(search_query, max_results=2):
    if not search_query.strip():
        raise ValueError("Search query is empty. Cannot proceed with an empty query.")

    search_response = exa.search_and_contents(
        search_query, use_autoprompt=True, text={"max_characters": 1000}, num_results=max_results, start_crawl_date=today, exclude_domains=["www.youtube.com"]
    )

    urls = [result.url for result in search_response.results]
    return urls, search_response.results

# Function to summarize the content of the URLs
def summarize_content(content):
    messages = [
        {"role": "user", "content": "Summarize the news content so that no important details are omitted. And from this summary I can fact check another news whether the other news is fake or not. Just give the summarized text no extra information or auto generated text to be present in output " + content}
    ]
    return make_llm_api_call(messages)

# Function to compute semantic similarity between original news and summarized news
def compute_similarity(original_news, summarized_news):
    messages = [
        {"role": "user", "content": f"Compare the following texts and provide the crediability score, that is whether the original news is true after comparing it with the information in the summarized news. Propotional to the fake news in original text reduce the credibility score also (that is if half the news is fake in original text then credibility score = 50). Give the credibility score as an INTEGER between 0 and 100 (no decimal point should be there), where 100 indicates that the news is completely true and 0 means the news is completely false. just give the credibility score as output and nothing else.:\n\nOriginal News: {original_news}\n\nSummarized News: {summarized_news}"}
    ]

    similarity_score = ""

    while True:
        similarity_score = make_llm_api_call(messages)
        print(f"Similarity Score Response: {similarity_score}")

        try:
            score = int(similarity_score.strip())
            if 0 <= score <= 100:
                return score
        except (IndexError, ValueError) as e:
            print(f"Error extracting credibility score: {e}")

    return 0  # Should never reach here if the while loop continues until a valid score is found

# Function to generate and print discrepancies
def find_discrepancies(original_news, summarized_news):
    messages = [
        {"role": "user", "content": f"Compare the following texts and identify any fact that is wrong in the original news compared to the real news.Just print the corrrect version of those facts and nothing else.Give the output in the form of sentences in a paragraph (containing maximum of 5 sentences.No auto generated text or bullet points/numbering of the sentence to be present in the paragraph.:\n\nOriginal News: {original_news}\n\nReal News: {summarized_news}"}
    ]
    discrepancies = make_llm_api_call(messages)
    return discrepancies

# Function to extract news content from a URL

def check_news_credibility(news_input):
    # Check if the input is a URL or not
    if re.match(r'http[s]?://', news_input):
        response = requests.get(news_input, headers={'User-Agent': user_agent})
        if response.status_code == 200:
            article = Article(news_input)
            article.set_html(response.text)
            article.parse()
            news_content = article.text
        else:
            raise Exception(f"Failed to download article with status code: {response.status_code}")
    else:
        news_content = news_input

    search_query = ""
    while not search_query.strip():
        search_query = get_search_query(news_content)
        print(f"Generated Search Query: {search_query}")

    urls, results = search_web(search_query)

    summarized_contents = []
    for result in results:
        summarized_content = summarize_content(result.text)
        while not summarized_content:
            summarized_content = summarize_content(result.text)
        summarized_contents.append(summarized_content)

    combined_summaries = " ".join(summarized_contents)
    credibility_score = compute_similarity(news_content, combined_summaries)
    discrepancies = find_discrepancies(news_content, combined_summaries)
    while not discrepancies:
        discrepancies = find_discrepancies(news_content, combined_summaries)

    return credibility_score, urls, discrepancies

nltk.download('stopwords')
# Preprocessing function to clean the text
def preprocess_text(text):
    if isinstance(text, float):  # Check if the text is a float (missing values)
        return ""
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing whitespace
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load the meta-model and vectorizer
mm_model = joblib.load('mm.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict using the meta-model
def predict_with_mm(news_content):
    text = preprocess_text(news_content)
    text_tfidf = vectorizer.transform([text])
    prediction_prob = mm_model.predict_proba(text_tfidf)
    return prediction_prob[0][1] * 100  # Assuming label 1 is for 'Real' news

# Function to simulate some processing
def process_text(text):
    news_input = text

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Run LLM and meta-model in parallel
        future_llm = executor.submit(check_news_credibility, news_input)
        future_mm = executor.submit(predict_with_mm, news_input)

        credibility_score_llm, sources, discrepancies = future_llm.result()
        mm_score = future_mm.result()

    # Combine the scores and convert to an integer
    final_score = int(mm_score * 0.1 + credibility_score_llm * 0.9)

    print(f"Credibility Score: {credibility_score_llm}")
    print(f"Meta-Model Score: {mm_score}")
    print(f"Final Combined Score: {final_score}")
    print("The sources are: ")
    for source in sources:
        print(source)
    print("Discrepancies: ")
    print(discrepancies)
    return {
        'num_letters': final_score,
        'source1': sources[0],
        'source2': sources[1] if len(sources) > 1 else None,
        'source3': discrepancies
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    text = data.get('text', '')

    # Using ThreadPoolExecutor to run process_text function asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_text, text)
        results = future.result()  # Get results once the task is completed

    return jsonify(results)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)