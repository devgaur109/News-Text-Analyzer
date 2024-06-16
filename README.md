# News-Text-Analyzer
### Overview of the Project

The "News Text Analyzer" project is a web application designed to analyze news snippets or links to determine their credibility. The application is built using Flask for the backend and HTML/CSS for the frontend. It leverages various technologies and APIs to assess the trustworthiness of the provided text.

### Technologies Used

1. **Flask**: A lightweight WSGI web application framework used to build the backend of the application.
2. **Newspaper3k**: A Python library used for extracting and parsing articles from news URLs.
3. **Exa**: An API service utilized for performing web searches.
4. **AwanLLM**: A large language model API used to extract main information and summarize news articles.
5. **Concurrent Futures**: A module to handle asynchronous tasks in the application.
6. **HTML/CSS**: For creating and styling the frontend of the application.

### Credibility Calculation

The credibility calculation process involves several steps:

1. **Extracting News Content**:
   - The application first checks if the input is a URL or plain text.
   - If it's a URL, the content is fetched using the `requests` library and parsed using `newspaper.Article`.

2. **Generating Search Query**:
   - The extracted news content is summarized into a search query using the AwanLLM API.
   - This query is designed to be a concise representation of the news that can be used for further searches.

3. **Searching the Web**:
   - The summarized search query is then used to search the web using the Exa API.
   - The search results are retrieved and processed.

4. **Summarizing Search Results**:
   - The content of each search result is summarized again using the AwanLLM API to create concise versions of the search results.

5. **Computing Similarity**:
   - The original news content and the summarized search results are compared to compute a similarity score.
   - This score represents how much of the original content matches with the summarized, trustworthy sources found in the search.

6. **Finding Discrepancies**:
   - Any discrepancies or fake news elements in the original content are identified by comparing it with the summarized search results using the AwanLLM API.

7. **Final Credibility Score**:
   - The credibility score is computed as an integer between 0 and 100, where 100 indicates completely true news, and 0 indicates completely false news.
   - The score is based on the proportion of fake elements found in the original content.

### Instructions to Run the Application

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Set Up API Keys**:
   Replace the placeholder API keys in `app.py` with your actual API keys for Exa and AwanLLM.
   To generate API keys, create an account on `https://exa.ai/` and `https://www.awanllm.com/` and navigate to their respective API keys tab.

4. **Run the Application**:
   Start the Flask server by opening the terminal in the cloned directory and writing:
   ```bash
   python app.py
   ```

5. **Access the Application**:
   Open your web browser and navigate to `http://127.0.0.1:5000/` to access the News Text Analyzer interface.
