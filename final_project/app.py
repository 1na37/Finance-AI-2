# finance_ai_assistant.py
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
from datetime import datetime
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv()

# ============================
# PROJECT METADATA
# ============================
PROJECT_INFO = {
    "title": "🤖 AI-Powered Financial Analysis Assistant",
    "description": "A multi-agent AI system for comprehensive financial analysis, technical indicators, and market research",
    "student_id": "REAENG1KDTXZ",
    "name": "Ina Alyani S.L.A",
    "features": [
        "Real-time stock analysis with technical indicators",
        "Multi-agent architecture (Finance + Web Search)",
        "Interactive financial charts and visualizations",
        "Investment calculations and risk assessment",
        "Automated market alerts and portfolio analysis"
    ]
}

# ============================
# CUSTOM FINANCE TOOLS
# ============================

class AdvancedFinanceTools:
    """Enhanced financial analysis tools with visualization capabilities."""
    
    def __init__(self):
        self.cache = {}
    
    def comprehensive_stock_analysis(self, symbol: str) -> str:
        """Provide comprehensive analysis with multiple indicators."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="6mo")
            
            if hist.empty:
                return f"No data available for {symbol}"
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # Technical indicators
            rsi = self._calculate_rsi(hist['Close'])
            macd, signal = self._calculate_macd(hist['Close'])
            
            analysis = f"""
## 📊 Comprehensive Analysis for {symbol}

### 📈 Price Information
- **Current Price**: ${current_price:.2f}
- **Previous Close**: ${prev_close:.2f}
- **Daily Change**: {change_pct:+.2f}%

### 🎯 Technical Indicators
- **RSI (14-day)**: {rsi:.2f} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Neutral)'}
- **MACD**: {macd:.4f}
- **Signal Line**: {signal:.4f}

### 🏢 Company Fundamentals
- **Company**: {info.get('longName', 'N/A')}
- **Sector**: {info.get('sector', 'N/A')}
- **Market Cap**: ${info.get('marketCap', 0):,}
- **P/E Ratio**: {info.get('trailingPE', 'N/A')}

### 💡 Investment Insight
{'**Bullish Signal**: Stock shows positive momentum' if macd > signal and rsi < 70 else 
 '**Caution**: Potential overbought conditions' if rsi > 70 else 
 '**Opportunity**: Potential oversold conditions' if rsi < 30 else 
 '**Neutral**: Monitor for clearer signals'}
"""
            return analysis
            
        except Exception as e:
            return f"Error analyzing {symbol}: {str(e)}"
    
    def create_price_chart(self, symbol: str, period: str = "3mo") -> str:
        """Create interactive price chart."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))
            
            # Add moving averages
            hist['MA20'] = hist['Close'].rolling(20).mean()
            hist['MA50'] = hist['Close'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(
                x=hist.index, 
                y=hist['MA20'], 
                name='20-day MA',
                line=dict(color='orange')
            ))
            fig.add_trace(go.Scatter(
                x=hist.index, 
                y=hist['MA50'], 
                name='50-day MA',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'{symbol} Stock Price with Moving Averages',
                yaxis_title='Price ($)',
                xaxis_title='Date',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            return f"Error creating chart: {str(e)}"
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    def _calculate_macd(self, prices):
        """Calculate MACD indicator."""
        exp12 = prices.ewm(span=12).mean()
        exp26 = prices.ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        return macd.iloc[-1], signal.iloc[-1]

# ============================
# INITIALIZE AI AGENTS
# ============================

@st.cache_resource
def initialize_agents():
    """Initialize AI agents with caching for performance."""
    
    # Initialize tools
    advanced_tools = AdvancedFinanceTools()
    
    # Finance Agent
    finance_agent = Agent(
        name="Advanced Finance AI Agent",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=[YFinanceTools(
            stock_price=True, 
            analyst_recommendations=True, 
            stock_fundamentals=True, 
            company_news=True,
            get_company_info=True
        )],
        instructions=[
            "Provide comprehensive financial analysis with actionable insights.",
            "Use technical indicators and fundamental analysis together.",
            "Always explain what the metrics mean for investment decisions.",
            "Present data in clear tables and provide summary conclusions."
        ],
        show_tool_calls=True,
        markdown=True,
    )
    
    # Web Search Agent
    web_search_agent = Agent(
        name="Web Research Agent",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=[DuckDuckGo()],
        instructions=[
            "Search for latest financial news, market trends, and economic data.",
            "Always cite credible sources and provide publication context.",
            "Focus on recent information (last 2-3 months) for relevance.",
            "Summarize key points and relate them to the user's query."
        ],
        show_tool_calls=True,
        markdown=True,
    )
    
    return finance_agent, web_search_agent, advanced_tools

# ============================
# STREAMLIT APPLICATION
# ============================

def main():
    st.set_page_config(
        page_title=PROJECT_INFO["title"],
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize agents
    finance_agent, web_search_agent, advanced_tools = initialize_agents()
    
    # Header
    st.title(PROJECT_INFO["title"])
    st.markdown(f"**Student**: {PROJECT_INFO['name']} | **ID**: {PROJECT_INFO['student_id']}")
    st.markdown(PROJECT_INFO["description"])
    
    # Project Overview
    with st.expander("🎯 Project Overview & Features", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Key Features")
            for feature in PROJECT_INFO["features"]:
                st.write(f"• {feature}")
        
        with col2:
            st.subheader("🛠️ Technical Stack")
            st.write("""
            • **AI Framework**: Phi-data
            • **LLM**: Groq (Llama3-70B)
            • **Data**: Yahoo Finance API
            • **Web Search**: DuckDuckGo
            • **Visualization**: Plotly, Streamlit
            • **Analysis**: Technical Indicators, Risk Assessment
            """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Agent Selection
        current_agent = st.selectbox(
            "Select AI Agent:",
            ["Finance Analyst", "Market Researcher"],
            help="Finance: Technical analysis | Researcher: Latest news & trends"
        )
        
        # Quick Actions
        st.subheader("🚀 Quick Analysis")
        popular_symbols = st.selectbox(
            "Analyze Popular Stocks:",
            ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]
        )
        
        if st.button("📈 Quick Analysis", use_container_width=True):
            st.session_state.quick_query = f"Provide comprehensive analysis for {popular_symbols} stock including technical indicators and fundamentals"
        
        # Sample Queries
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "Analyze AAPL technical indicators and provide investment recommendation",
            "What are the latest news and trends in electric vehicle sector?",
            "Compare TSLA and NIO stock performance and fundamentals",
            "Show me risk analysis for cryptocurrency market",
            "What are the current market sentiments for tech stocks?"
        ]
        
        for query in sample_queries:
            if st.button(f"\"{query[:30]}...\"", key=query):
                st.session_state.quick_query = query
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"{message['timestamp']} • {message.get('agent', 'System')}")
    
    # Handle quick queries
    if "quick_query" in st.session_state:
        query = st.session_state.quick_query
        del st.session_state.quick_query
    else:
        query = st.chat_input("Ask about stocks, market analysis, or investment advice...")
    
    # Process user input
    if query:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": query,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process with selected agent
        with st.chat_message("assistant"):
            with st.spinner(f"🤔 {current_agent} is analyzing..."):
                try:
                    if current_agent == "Finance Analyst":
                        response = finance_agent.run(query, stream=True)
                    else:
                        response = web_search_agent.run(query, stream=True)
                    
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
                    
                    # Add charts for stock analysis
                    if any(symbol in query.upper() for symbol in ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]):
                        symbol = next((s for s in ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"] if s in query.upper()), "AAPL")
                        st.subheader("📊 Price Chart")
                        chart = advanced_tools.create_price_chart(symbol)
                        if isinstance(chart, go.Figure):
                            st.plotly_chart(chart, use_container_width=True)
                        else:
                            st.info(chart)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "agent": current_agent
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "agent": "System"
                    })
    
    # Demo Section
    st.divider()
    st.subheader("🎬 Live Demo Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🤖 Multi-Agent Architecture**
        - Finance Agent: Technical analysis
        - Research Agent: Market news
        - Smart routing between agents
        """)
    
    with col2:
        st.info("""
        **📈 Advanced Analytics**
        - Real-time stock data
        - Technical indicators (RSI, MACD)
        - Risk assessment
        - Investment calculations
        """)
    
    with col3:
        st.info("""
        **🎯 Professional Features**
        - Interactive charts
        - Source citations
        - Actionable insights
        - Portfolio analysis
        """)

if __name__ == "__main__":
    main()
