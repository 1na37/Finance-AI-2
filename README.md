# Multi-AI Financial Assistant

This project is a multi-agent AI system that combines a web search agent and a finance agent to answer questions about stocks and companies.

## Features

- **Web Search Agent**: Uses DuckDuckGo to search the web for relevant information and cites sources.
- **Finance Agent**: Uses Yahoo Finance to get stock prices, analyst recommendations, fundamentals, and company news.
- **Multi-Agent**: Both agents work together to provide comprehensive answers.

## How to Run

1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Set up environment variables in a `.env` file:
   - `PHI_API_KEY`
   - `GROQ_API_KEY`
4. Run the Streamlit app: `streamlit run app.py`

