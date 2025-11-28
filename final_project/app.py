import streamlit as st
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for financial information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=[
        "Use web search to find relevant financial information.",
        "Always cite your sources in the response.",
        "Provide accurate and up-to-date information."
    ],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(
        stock_price=True, 
        analyst_recommendations=True, 
        stock_fundamentals=True, 
        company_news=True, 
        get_company_info=True, 
        historical_stock_prices=True, 
        technical_indicators=True
    )],
    instructions=[
        "Use tables to display financial data when appropriate.",
        "Provide clear analysis of financial information.",
        "Be precise with numbers and percentages."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Page configuration
st.set_page_config(
    page_title="Finance AI Assistant", 
    page_icon="💸", 
    layout="wide"
)

st.title("💸 Finance AI Assistant")
st.caption("Multi-agent financial analysis suite with web search capabilities")

# ------------------------------
# Session state initialization
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_agent" not in st.session_state:
    st.session_state.current_agent = "Finance Agent"

# ------------------------------
# Sidebar UI
# ------------------------------
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    
    # Agent selection
    st.session_state.current_agent = st.selectbox(
        "Select Agent:",
        options=["Finance Agent", "Web Search Agent"],
        help="Choose which agent to use for your query"
    )
    
    # Agent descriptions
    if st.session_state.current_agent == "Finance Agent":
        st.info("""
        **Finance Agent**: 
        - Stock prices and analysis
        - Company fundamentals
        - Technical indicators
        - Historical data
        - Analyst recommendations
        """)
    else:
        st.info("""
        **Web Search Agent**:
        - Latest financial news
        - Market research
        - Economic data
        - Company information
        """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")

# ------------------------------
# Display chat messages
# ------------------------------
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if "timestamp" in message:
            st.caption(f"Agent: {message.get('agent', 'N/A')} • {message['timestamp']}")

# ------------------------------
# Chat input and processing
# ------------------------------
if prompt := st.chat_input("Ask about stocks, companies, or financial news..."):
    # Add user message to chat history
    user_timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "timestamp": user_timestamp
    })
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
        st.caption(f"Sent at: {user_timestamp}")
    
    # Process with selected agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(f"Consulting {st.session_state.current_agent}..."):
            try:
                if st.session_state.current_agent == "Finance Agent":
                    response = finance_agent.run(prompt, stream=True)
                else:
                    response = web_search_agent.run(prompt, stream=True)
                
                # Display streaming response
                response_container = st.empty()
                full_response = ""
                
                for chunk in response:
                    if hasattr(chunk, 'content'):
                        full_response += chunk.content
                    else:
                        full_response += str(chunk)
                    response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)
                
                # Add assistant response to chat history
                assistant_timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": assistant_timestamp,
                    "agent": st.session_state.current_agent
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent": st.session_state.current_agent
                })

# ------------------------------
# Quick action buttons
# ------------------------------
st.divider()
st.subheader("🚀 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📈 Stock Price", use_container_width=True):
        st.chat_input("Ask about stock price...", value="What is the current stock price of AAPL?")

with col2:
    if st.button("📊 Fundamentals", use_container_width=True):
        st.chat_input("Ask about fundamentals...", value="Show me the fundamentals for Tesla")

with col3:
    if st.button("📰 Latest News", use_container_width=True):
        st.chat_input("Ask for news...", value="Find latest news about NVIDIA")

with col4:
    if st.button("💡 Analysis", use_container_width=True):
        st.chat_input("Ask for analysis...", value="Technical analysis for Amazon stock")

# ------------------------------
# Footer
# ------------------------------
st.divider()
st.caption("""
💡 **Tips**: 
- Use specific ticker symbols (AAPL, TSLA, etc.) for better results
- Ask for "analyst recommendations" or "technical indicators"
- Switch to Web Search Agent for latest market news
""")
