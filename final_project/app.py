import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os

# Load environment variables (if using .env)
from dotenv import load_dotenv
load_dotenv()

# Set up the agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information you need",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=[
        "Use web search to find relevant information.",
        "Cite your sources in the response."
    ],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, 
        stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources in your answers", "Work together to provide the best answer", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Streamlit app
st.title("Multi-AI Financial Assistant")
st.write("This app uses a multi-agent AI system to answer your questions about stocks and finance.")

query = st.text_input("Enter your question about a company or stock:", value="Summarize analyst recommendations and share the latest news for Tesla Inc.")

if st.button("Get Answer"):
    if query:
        with st.spinner("Thinking..."):
            response = multi_ai_agent.run(query)
            st.markdown(response)
    else:
        st.warning("Please enter a question.")
