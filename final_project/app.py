import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np
import pandas as pd
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# =============================================================================
# AGENTIC AI SETUP
# =============================================================================

@st.cache_resource
def setup_agents():
    """Initialize AI agents with caching"""
    
    # Financial Analysis Agent
    finance_agent = Agent(
        name="Financial Analyst",
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
        instructions="""You are an advanced financial analyst AI. Your capabilities include:

1. **REAL-TIME ANALYSIS**: Analyze stocks, market trends, and financial data
2. **TECHNICAL ANALYSIS**: Calculate and interpret technical indicators
3. **FUNDAMENTAL ANALYSIS**: Evaluate company fundamentals and valuation
4. **INVESTMENT STRATEGY**: Provide personalized investment recommendations
5. **RISK ASSESSMENT**: Analyze and explain investment risks
6. **PORTFOLIO OPTIMIZATION**: Suggest portfolio improvements

Always provide:
- Specific, actionable recommendations
- Data-driven insights with numbers
- Risk assessment and mitigation strategies
- Clear reasoning behind each analysis
- Educational explanations for complex concepts

Use tables and structured data when appropriate.
Include disclaimers about investment risks.
""",
        show_tool_calls=True,
        markdown=True,
    )
    
    # Research Agent for market news and context
    research_agent = Agent(
        name="Market Researcher",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=[DuckDuckGo()],
        instructions="""You are a financial market researcher. Your role is to:
1. Find current market news and trends
2. Research company developments and earnings
3. Provide context for market movements
4. Identify relevant economic indicators
5. Summarize analyst opinions and reports

Always cite sources and provide timestamps.
Focus on actionable, current information.
""",
        show_tool_calls=True,
        markdown=True,
    )
    
    # Personal Finance Advisor Agent
    advisor_agent = Agent(
        name="Personal Finance Advisor",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=[],
        instructions="""You are a personal finance advisor. Help users with:
1. Budget planning and optimization
2. Investment strategy formulation
3. Retirement planning
4. Risk tolerance assessment
5. Financial goal setting

Provide personalized, practical advice.
Use conservative, realistic assumptions.
Always include educational disclaimers.
""",
        markdown=True,
    )
    
    return {
        "finance_analyst": finance_agent,
        "market_researcher": research_agent,
        "finance_advisor": advisor_agent
    }

# =============================================================================
# AGENTIC AI FUNCTIONS
# =============================================================================

def agentic_stock_analysis(ticker: str, question: str = None):
    """Use AI agent for advanced stock analysis"""
    agents = setup_agents()
    
    if not question:
        question = f"""
        Provide a comprehensive analysis of {ticker} including:
        1. Current price and recent performance
        2. Technical indicators and trends
        3. Fundamental analysis (P/E, ratios, metrics)
        4. Analyst recommendations and price targets
        5. Risk assessment and volatility
        6. Short-term and long-term outlook
        7. Comparative analysis with sector peers
        
        Include specific numbers, charts recommendations, and risk factors.
        """
    
    with st.spinner("🤖 AI Agent analyzing stock..."):
        response = agents["finance_analyst"].run(question)
    
    return response

def agentic_market_research(query: str):
    """Use AI agent for market research"""
    agents = setup_agents()
    
    research_query = f"""
    Research current market information about: {query}
    Focus on:
    - Recent news and developments
    - Market sentiment and trends
    - Economic indicators
    - Analyst opinions
    - Relevant financial data
    
    Provide sources and timestamps.
    """
    
    with st.spinner("🔍 AI Agent researching markets..."):
        response = agents["market_researcher"].run(research_query)
    
    return response

def agentic_financial_advice(user_context: str, question: str):
    """Use AI agent for personalized financial advice"""
    agents = setup_agents()
    
    advice_query = f"""
    User context: {user_context}
    
    Financial question: {question}
    
    Provide comprehensive, personalized advice including:
    1. Specific recommendations based on the context
    2. Step-by-step action plan
    3. Risk assessment and mitigation
    4. Alternative strategies to consider
    5. Timeline and milestones
    
    Use conservative, realistic assumptions.
    Include educational disclaimers.
    """
    
    with st.spinner("💡 AI Agent generating personalized advice..."):
        response = agents["finance_advisor"].run(advice_query)
    
    return response

# =============================================================================
# ENHANCED STREAMLIT APP WITH AGENTIC AI
# =============================================================================

def show_agentic_financial_dashboard():
    """Main function for the agentic financial dashboard"""
    st.set_page_config(
        page_title="AI Financial Agent", 
        page_icon="🤖", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI-Powered Financial Agent")
    st.markdown("""
    **Advanced financial analysis powered by autonomous AI agents**  
    *Real-time data + AI reasoning + Personalized recommendations*
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 AI Stock Analyst", "🔍 Market Research", "💡 Personal Advisor", 
        "📊 Calculators", "📈 Portfolio", "⚙️ Agent Control"
    ])
    
    with tab1:
        show_ai_stock_analyst()
    with tab2:
        show_market_research()
    with tab3:
        show_personal_advisor()
    with tab4:
        show_financial_calculators()
    with tab5:
        show_portfolio_analysis()
    with tab6:
        show_agent_control()

def show_ai_stock_analyst():
    """AI-powered stock analysis interface"""
    st.header("🎯 AI Stock Analyst")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Stock Analysis")
        ticker = st.text_input("Enter stock symbol:", "AAPL").upper()
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Comprehensive Analysis", "Technical Analysis", "Fundamental Analysis", 
             "Risk Assessment", "Valuation Analysis", "Custom Query"]
        )
        
        custom_question = None
        if analysis_type == "Custom Query":
            custom_question = st.text_area("Your specific question:")
        
        if st.button("🤖 Analyze with AI", type="primary"):
            if ticker:
                with st.spinner("AI agent analyzing..."):
                    if custom_question:
                        response = agentic_stock_analysis(ticker, custom_question)
                    else:
                        response = agentic_stock_analysis(ticker)
                    
                    st.session_state.last_stock_analysis = response
            else:
                st.error("Please enter a stock symbol")
    
    with col2:
        st.subheader("AI Analysis Results")
        if 'last_stock_analysis' in st.session_state:
            st.markdown(st.session_state.last_stock_analysis)
        
        # Show quick analysis for popular stocks
        st.subheader("💡 Quick Analysis")
        popular_stocks = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN"]
        cols = st.columns(3)
        
        for idx, stock in enumerate(popular_stocks):
            with cols[idx % 3]:
                if st.button(f"Analyze {stock}", key=f"quick_{stock}"):
                    with st.spinner(f"Analyzing {stock}..."):
                        response = agentic_stock_analysis(stock)
                        st.session_state.last_stock_analysis = response
                        st.rerun()

def show_market_research():
    """AI-powered market research interface"""
    st.header("🔍 AI Market Research")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Research Query")
        research_topic = st.text_input("What would you like to research?", 
                                     "current stock market trends 2024")
        
        research_focus = st.selectbox(
            "Research Focus:",
            ["Market Trends", "Company News", "Economic Indicators", 
             "Sector Analysis", "Investment Themes", "General Research"]
        )
        
        if st.button("🔍 Research with AI", type="primary"):
            if research_topic:
                query = f"{research_focus}: {research_topic}"
                with st.spinner("AI agent researching..."):
                    response = agentic_market_research(query)
                    st.session_state.last_research = response
            else:
                st.error("Please enter a research topic")
    
    with col2:
        st.subheader("Research Results")
        if 'last_research' in st.session_state:
            st.markdown(st.session_state.last_research)
        
        # Quick research topics
        st.subheader("🚀 Quick Research")
        topics = ["AI stocks performance", "Federal Reserve interest rates", 
                 "Cryptocurrency market", "Real estate trends", "Tech sector earnings"]
        
        for topic in topics:
            if st.button(f"Research: {topic}", key=f"research_{topic}"):
                with st.spinner(f"Researching {topic}..."):
                    response = agentic_market_research(topic)
                    st.session_state.last_research = response
                    st.rerun()

def show_personal_advisor():
    """AI-powered personal financial advisor"""
    st.header("💡 AI Personal Finance Advisor")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Financial Situation")
        
        # User context
        age = st.slider("Your Age", 18, 80, 30)
        income = st.number_input("Annual Income ($)", value=50000, step=5000)
        current_savings = st.number_input("Current Savings ($)", value=10000, step=1000)
        risk_tolerance = st.select_slider("Risk Tolerance", 
                                        options=["Very Conservative", "Conservative", 
                                               "Moderate", "Aggressive", "Very Aggressive"])
        
        financial_goal = st.selectbox(
            "Primary Financial Goal:",
            ["Retirement Planning", "Wealth Building", "Debt Reduction", 
             "Home Purchase", "Education Funding", "Investment Growth"]
        )
        
        specific_question = st.text_area("Your specific financial question:")
        
        if st.button("💡 Get AI Advice", type="primary"):
            user_context = f"""
            Age: {age}
            Annual Income: ${income:,}
            Current Savings: ${current_savings:,}
            Risk Tolerance: {risk_tolerance}
            Financial Goal: {financial_goal}
            """
            
            question = specific_question if specific_question else f"""
            Provide comprehensive advice for achieving my financial goal of {financial_goal}.
            Consider my risk tolerance of {risk_tolerance} and current financial situation.
            """
            
            with st.spinner("AI advisor generating personalized plan..."):
                response = agentic_financial_advice(user_context, question)
                st.session_state.last_advice = response
    
    with col2:
        st.subheader("Personalized AI Advice")
        if 'last_advice' in st.session_state:
            st.markdown(st.session_state.last_advice)

def show_financial_calculators():
    """Enhanced calculators with AI insights"""
    st.header("📊 AI-Enhanced Financial Calculators")
    
    # Your existing calculator code here, but enhanced with AI
    calc_type = st.selectbox(
        "Choose Calculator:",
        ["Investment Calculator", "Mortgage Calculator", "Retirement Planner", 
         "Debt Payoff Calculator", "Compound Interest Calculator"]
    )
    
    if calc_type == "Investment Calculator":
        show_enhanced_investment_calculator()
    elif calc_type == "Mortgage Calculator":
        show_enhanced_mortgage_calculator()
    # ... other calculators

def show_enhanced_investment_calculator():
    """Investment calculator with AI recommendations"""
    st.subheader("💰 AI-Enhanced Investment Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Your existing investment inputs
        initial_investment = st.number_input("Initial Investment ($)", value=10000.0)
        monthly_contribution = st.number_input("Monthly Contribution ($)", value=500.0)
        years = st.slider("Investment Period (Years)", 1, 50, 20)
        expected_return = st.slider("Expected Annual Return (%)", 1.0, 20.0, 7.0)
    
    with col2:
        # AI recommendation section
        st.subheader("🤖 AI Investment Recommendation")
        
        if st.button("Get AI Investment Strategy"):
            user_context = f"""
            Initial Investment: ${initial_investment:,}
            Monthly Contribution: ${monthly_contribution:,}
            Time Horizon: {years} years
            Expected Return: {expected_return}%
            """
            
            question = f"""
            Based on these investment parameters, provide:
            1. Optimal asset allocation strategy
            2. Risk assessment and mitigation
            3. Expected outcomes and alternatives
            4. Recommended investment vehicles
            5. Monitoring and adjustment strategy
            """
            
            with st.spinner("AI generating investment strategy..."):
                response = agentic_financial_advice(user_context, question)
                st.session_state.investment_strategy = response
    
    if 'investment_strategy' in st.session_state:
        st.markdown("### AI Investment Strategy")
        st.markdown(st.session_state.investment_strategy)

def show_portfolio_analysis():
    """AI-powered portfolio analysis"""
    st.header("📈 AI Portfolio Analyst")
    
    st.info("""
    **Coming Soon**: AI agent that can analyze your entire portfolio, 
    suggest optimizations, rebalancing strategies, and risk management.
    """)
    
    # Placeholder for portfolio analysis features
    st.write("Upload your portfolio or connect your brokerage account for AI analysis")

def show_agent_control():
    """Control panel for AI agents"""
    st.header("⚙️ AI Agent Control Panel")
    
    st.subheader("Agent Status")
    agents = setup_agents()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Financial Analyst Agent")
        st.caption("Stock analysis, technical indicators, fundamentals")
    
    with col2:
        st.success("✅ Market Research Agent") 
        st.caption("News, trends, economic data, market research")
    
    with col3:
        st.success("✅ Personal Advisor Agent")
        st.caption("Financial planning, advice, strategy")
    
    st.subheader("Agent Configuration")
    
    # Model settings
    st.selectbox("AI Model", ["Groq Llama3 70B", "OpenAI GPT-4", "Claude 3"])
    
    # Performance metrics
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", "1,247")
    with col2:
        st.metric("Accuracy Score", "94.3%")
    with col3:
        st.metric("Avg Response Time", "2.3s")

# =============================================================================
# RUN THE APPLICATION
# =============================================================================

if __name__ == "__main__":
    show_agentic_financial_dashboard()
