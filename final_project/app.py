import streamlit as st
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
import os
from dotenv import load_dotenv
from datetime import datetime
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import time

# Load environment variables
load_dotenv()

# ============================
# CUSTOM FINANCE TOOLS
# ============================

class IndicatorTools:
    """Tools for calculating technical indicators like RSI, MACD, and Moving Averages."""
    
    def calculate_rsi(self, symbol: str, period: str = "3mo") -> str:
        """Calculate RSI for a stock symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            prices = data['Close']
            
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
            return f"RSI for {symbol}: {current_rsi:.2f}\n\nRSI Levels:\n- Overbought: >70\n- Oversold: <30\n- Current: {current_rsi:.2f}"
        except Exception as e:
            return f"Error calculating RSI for {symbol}: {str(e)}"

    def calculate_macd(self, symbol: str, period: str = "6mo") -> str:
        """Calculate MACD for a stock symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            prices = data['Close']
            
            exp12 = prices.ewm(span=12, adjust=False).mean()
            exp26 = prices.ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            current_macd = macd.iloc[-1] if not macd.empty else 0
            current_signal = signal.iloc[-1] if not signal.empty else 0
            current_histogram = histogram.iloc[-1] if not histogram.empty else 0
            
            return f"""MACD for {symbol}:
- MACD Line: {current_macd:.4f}
- Signal Line: {current_signal:.4f}
- Histogram: {current_histogram:.4f}

Interpretation:
- MACD > Signal: Bullish
- MACD < Signal: Bearish
- Histogram > 0: Bullish momentum
- Histogram < 0: Bearish momentum"""
        except Exception as e:
            return f"Error calculating MACD for {symbol}: {str(e)}"

    def moving_averages(self, symbol: str, period: str = "1y") -> str:
        """Calculate moving averages for a stock symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            prices = data['Close']
            
            ma20 = prices.rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else 0
            ma50 = prices.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else 0
            current_price = prices.iloc[-1] if not prices.empty else 0
            
            return f"""Moving Averages for {symbol}:
- Current Price: ${current_price:.2f}
- 20-day MA: ${ma20:.2f}
- 50-day MA: ${ma50:.2f}

Trend Analysis:
- Price > MA20 & MA50: Uptrend
- Price < MA20 & MA50: Downtrend
- MA20 > MA50: Bullish crossover
- MA20 < MA50: Bearish crossover"""
        except Exception as e:
            return f"Error calculating moving averages for {symbol}: {str(e)}"

class FinanceAPI:
    """Tools for fetching stock data and exchange rates."""
    
    def fetch_stock_data(self, symbol: str, period: str = "1mo") -> str:
        """Fetch comprehensive stock data."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period=period)
            
            if hist.empty:
                return f"No data found for symbol: {symbol}"
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            return f"""Stock Data for {symbol}:
- Current Price: ${current_price:.2f}
- Previous Close: ${prev_close:.2f}
- Change: ${change:.2f} ({change_percent:+.2f}%)
- Day High: ${hist['High'].iloc[-1]:.2f}
- Day Low: ${hist['Low'].iloc[-1]:.2f}
- Volume: {hist['Volume'].iloc[-1]:,}

Company Info:
- Name: {info.get('longName', 'N/A')}
- Sector: {info.get('sector', 'N/A')}
- Market Cap: ${info.get('marketCap', 0):,}"""
        except Exception as e:
            return f"Error fetching data for {symbol}: {str(e)}"

    def get_exchange_rate(self, base: str = "USD", target: str = "IDR") -> str:
        """Get current exchange rate."""
        try:
            url = f"https://api.exchangerate.host/latest?base={base}&symbols={target}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if response.status_code == 200 and data.get('success', False):
                rate = data['rates'][target]
                return f"Exchange Rate: 1 {base} = {rate:.2f} {target}"
            else:
                return "Error fetching exchange rate data"
        except Exception as e:
            return f"Error getting exchange rate: {str(e)}"

class InvestmentTools:
    """Tools for investment calculations and analysis."""
    
    def compound_growth(self, principal: float, monthly_contribution: float, 
                       years: int, annual_return: float) -> str:
        """Calculate compound growth with regular contributions."""
        try:
            monthly_rate = annual_return / 12 / 100
            months = years * 12
            
            future_value = principal
            for month in range(months):
                future_value = future_value * (1 + monthly_rate) + monthly_contribution
            
            total_contributions = principal + (monthly_contribution * months)
            total_interest = future_value - total_contributions
            
            return f"""Investment Projection:
- Initial Investment: ${principal:,.2f}
- Monthly Contribution: ${monthly_contribution:,.2f}
- Investment Period: {years} years
- Annual Return: {annual_return}%

Results:
- Future Value: ${future_value:,.2f}
- Total Contributions: ${total_contributions:,.2f}
- Total Interest Earned: ${total_interest:,.2f}
- Growth Multiple: {future_value/total_contributions:.2f}x"""
        except Exception as e:
            return f"Error calculating compound growth: {str(e)}"

    def risk_analysis(self, symbol: str, period: str = "1y") -> str:
        """Perform basic risk analysis for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            prices = data['Close']
            
            returns = prices.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility %
            max_drawdown = (prices / prices.cummax() - 1).min() * 100
            
            return f"""Risk Analysis for {symbol}:
- Annualized Volatility: {volatility:.2f}%
- Maximum Drawdown: {max_drawdown:.2f}%
- Average Daily Return: {returns.mean()*100:.4f}%

Risk Assessment:
- Volatility < 20%: Low risk
- Volatility 20-40%: Medium risk  
- Volatility > 40%: High risk"""
        except Exception as e:
            return f"Error performing risk analysis for {symbol}: {str(e)}"

# ============================
# INITIALIZE AGENTS
# ============================

# Initialize tool instances
indicator_tools = IndicatorTools()
finance_api = FinanceAPI()
investment_tools = InvestmentTools()

# Finance Agent with custom tools
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        finance_api.fetch_stock_data,
        finance_api.get_exchange_rate,
        indicator_tools.calculate_rsi,
        indicator_tools.calculate_macd,
        indicator_tools.moving_averages,
        investment_tools.compound_growth,
        investment_tools.risk_analysis
    ],
    instructions=[
        "Use financial calculations and technical indicators to provide comprehensive analysis.",
        "Explain what each metric means for investment decisions.",
        "Use tables and clear formatting when presenting data.",
        "Provide actionable insights based on the analysis."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for financial information and news",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=[
        "Search for current financial news, market analysis, and economic data.",
        "Always cite your sources and provide timeliness context.",
        "Focus on credible financial sources and recent information."
    ],
    show_tool_calls=True,
    markdown=True,
)

# ============================
# STREAMLIT APP
# ============================

st.set_page_config(
    page_title="Advanced Finance AI Assistant", 
    page_icon="💹", 
    layout="wide"
)

st.title("💹 Advanced Finance AI Assistant")
st.caption("Professional financial analysis with technical indicators, investment tools, and market research")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_agent" not in st.session_state:
    st.session_state.current_agent = "Finance Agent"

# Sidebar
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    
    st.session_state.current_agent = st.selectbox(
        "Select Agent:",
        options=["Finance Agent", "Web Search Agent"],
        help="Finance Agent: Technical analysis & calculations | Web Search: Market news & research"
    )
    
    # Agent capabilities
    if st.session_state.current_agent == "Finance Agent":
        st.success("""
        **Finance Agent Capabilities:**
        📈 Technical Indicators (RSI, MACD, Moving Averages)
        💰 Stock Data & Fundamentals  
        📊 Investment Calculations
        🎯 Risk Analysis
        💱 Exchange Rates
        """)
    else:
        st.info("""
        **Web Search Agent:**
        🔍 Latest Financial News
        📰 Market Analysis
        🌍 Economic Data
        🏢 Company Research
        📑 Source Citations
        """)
    
    st.divider()
    
    # Quick analysis options
    st.subheader("🚀 Quick Analysis")
    
    if st.button("📈 Technical Analysis", use_container_width=True):
        st.session_state.quick_action = "technical"
    
    if st.button("💰 Investment Calc", use_container_width=True):
        st.session_state.quick_action = "investment"
    
    if st.button("📊 Risk Assessment", use_container_width=True):
        st.session_state.quick_action = "risk"
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")

# Display chat messages
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if "timestamp" in message:
            st.caption(f"Agent: {message.get('agent', 'N/A')} • {message['timestamp']}")

# Handle quick actions
if "quick_action" in st.session_state:
    quick_prompts = {
        "technical": "Please provide technical analysis for AAPL including RSI, MACD, and moving averages",
        "investment": "Calculate compound growth for $10,000 initial investment with $500 monthly contributions at 7% annual return over 20 years",
        "risk": "Perform risk analysis for TSLA stock including volatility and drawdown analysis"
    }
    
    prompt = quick_prompts.get(st.session_state.quick_action, "")
    del st.session_state.quick_action

# Chat input
if prompt := st.chat_input("Ask about stocks, technical analysis, or investments..."):
    # Add user message
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

# Quick stock lookup
st.divider()
st.subheader("🔍 Quick Stock Lookup")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("AAPL", use_container_width=True):
        st.chat_input("Quick lookup...", value="Show me current data and technical analysis for AAPL")

with col2:
    if st.button("TSLA", use_container_width=True):
        st.chat_input("Quick lookup...", value="Provide technical analysis and risk assessment for TSLA")

with col3:
    if st.button("GOOGL", use_container_width=True):
        st.chat_input("Quick lookup...", value="Analyze GOOGL stock with moving averages and RSI")

with col4:
    if st.button("USD/IDR", use_container_width=True):
        st.chat_input("Quick lookup...", value="What is the current USD to IDR exchange rate?")

# Footer with tool explanations
st.divider()
st.caption("""
💡 **Available Tools**: 
- **RSI**: Relative Strength Index (overbought/oversold)
- **MACD**: Moving Average Convergence Divergence (trend momentum)  
- **Moving Averages**: Support/Resistance levels
- **Compound Growth**: Future value calculations
- **Risk Analysis**: Volatility and drawdown metrics
- **Real-time Data**: Current prices and exchange rates
""")
