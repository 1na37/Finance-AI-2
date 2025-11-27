# app.py
import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import os
import re
from typing import Dict, List, Optional, Any, Tuple

# Data Analysis & Visualization imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For live stock/crypto prices (best-effort)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ------------------------------
# Validate API key before running
# ------------------------------
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("❌ OpenRouter API key not found. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# ------------------------------
# Enhanced Utilities
# ------------------------------
def clean_response(content: str) -> str:
    """Clean response content by removing markdown and extra whitespace."""
    if not content:
        return ""
    content = content.replace('```', '').replace('<s>', '').replace('</s>', '').strip()
    content = '\n'.join([line.rstrip() for line in content.split('\n') if line.strip()])
    return content

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Robust JSON extraction with multiple fallback strategies."""
    if not text:
        return None
    
    # Strategy 1: Find first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Strategy 2: Replace single quotes
            candidate = candidate.replace("'", '"')
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    
    # Strategy 3: Look for JSON-like patterns
    json_pattern = r'\{[^{}]*"[^{}]*"[^{}]*\}'
    matches = re.findall(json_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

def post_openrouter(model: str, messages: List[Dict], temperature: float = 0.4, 
                   max_tokens: int = 800, timeout: int = 45) -> Optional[str]:
    """Enhanced API call with better error handling."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Finance AI Agent"
    }
    
    try:
        response = requests.post(OPENROUTER_CHAT_URL, headers=headers, 
                               data=json.dumps(payload), timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content")
                return choice.get("text")
        
        # Enhanced error reporting
        error_msg = f"API Error {model}: {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f" - {error_data.get('error', {}).get('message', response.text[:100])}"
            except:
                error_msg += f" - {response.text[:100]}"
        st.error(error_msg)
        return None
        
    except requests.exceptions.Timeout:
        st.error(f"Request timeout for {model} after {timeout}s.")
        return None
    except Exception as e:
        st.error(f"OpenRouter request failed: {str(e)}")
        return None

# ------------------------------
# Enhanced Assistant Configs - FIXED SYNTAX ERROR
# ------------------------------
ASSISTANTS = {
    "💼 Personal Finance Advisor": {
        "primary": "qwen/qwen3-235b-a22b:free",
        "backup1": "deepseek/deepseek-chat-v3.1:free", 
        "backup2": "mistralai/mistral-small-3.2-24b-instruct:free",
        "system_prompt": """You are an autonomous Personal Finance Advisor agent. Your capabilities include:
- Analyzing income, expenses, and savings patterns
- Creating personalized budget plans
- Debt management strategies
- Emergency fund planning
- Basic investment guidance
- Data visualization and financial analysis

AGENTIC BEHAVIOR:
1. Always start with a clear plan
2. Use tools when numerical calculations are needed
3. Ask clarifying questions when information is missing
4. Provide step-by-step reasoning
5. Include specific, actionable recommendations
6. Use visualization tools to create charts and graphs

Always include: "I am not a licensed financial advisor; this is educational information."
""" # The closing """ is now clean
    },
    "📈 Investment Analyst": {
        "primary": "deepseek/deepseek-r1-0528:free",
        "backup1": "openai/gpt-oss-120b:free",
        "backup2": "qwen/qwen3-coder:free", 
        "system_prompt": """You are an autonomous Investment Analyst agent. Your capabilities include:
- Portfolio analysis and construction
- Risk assessment and management
- Investment return projections
- Asset allocation strategies
- Market research and analysis
- Technical analysis and charting

AGENTIC BEHAVIOR:
1. Analyze current portfolio when provided
2. Use tools for precise calculations
3. Consider risk tolerance and time horizon
4. Provide comparative analysis
5. Suggest diversification strategies
6. Create visualizations for investment analysis

Include: "Not professional financial advice. Past performance ≠ future results."
""" # The closing """ is now clean
    },
    "📊 Budget Planner": {
        "primary": "google/gemini-2.0-flash-exp:free",
        "backup1": "x-ai/grok-4-fast:free",
        "backup2": "qwen/qwen3-235b-a22b:free",
        "system_prompt": """You are an autonomous Budget Planner agent. Your capabilities include:
- Creating detailed budget plans
- Expense categorization and tracking
- Savings goal planning
- Cash flow optimization
- Financial habit formation
- Budget visualization and reporting

AGENTIC BEHAVIOR:
1. Build comprehensive budget frameworks
2. Identify saving opportunities
3. Set realistic financial goals
4. Provide monthly tracking guidance
5. Adjust plans based on user feedback
6. Create visual budget breakdowns

Include: "Educational budget planning only."
""" # The closing """ is now clean
    },
    "🏛️ Economic Researcher": {
        "primary": "google/gemini-2.0-flash-exp:free", 
        "backup1": "deepseek/deepseek-chat-v3.1:free",
        "backup2": "qwen/qwen3-235b-a22b:free",
        "system_prompt": """You are an autonomous Economic Researcher agent. Your capabilities include:
- Macroeconomic analysis
- Policy impact assessment
- Market trend research
- Economic indicator interpretation
- Research summarization
- Data visualization and trend analysis

AGENTIC BEHAVIOR:
1. Research current economic conditions
2. Analyze policy implications
3. Connect macroeconomic trends to personal finance
3. Provide data-driven insights
5. Source and verify information
6. Create economic data visualizations

Include: "Educational economic analysis only."
""" # The closing """ is now clean
    },
    "🧾 Tax Helper (General)": {
        "primary": "mistralai/mistral-small-3.2-24b-instruct:free",
        "backup1": "qwen/qwen3-235b-a22b:free",
        "backup2": "deepseek/deepseek-r1-0528:free",
        "system_prompt": """You are an autonomous Tax Helper agent. Your capabilities include:
- Tax estimation and planning
- Deduction identification
- Tax-advantaged account guidance
- Filing preparation overview
- Tax implication analysis
- Tax scenario visualization

AGENTIC BEHAVIOR:
1. Estimate tax liabilities accurately
2. Identify potential deductions
3. Explain tax concepts clearly
4. Provide filing preparation tips
5. Clarify jurisdiction limitations
6. Create tax comparison visualizations

Include: "General information only - consult a tax professional."
""" # The closing """ is now clean
    }
}

RELIABLE_MODELS = [
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-chat-v3.1:free", 
    "google/gemini-2.0-flash-exp:free",
    "x-ai/grok-4-fast:free",
    "mistralai/mistral-small-3.2-24b-instruct:free"
]

def get_assistant_model(assistant_name: str, attempt: int = 1) -> str:
    """Get model for assistant with fallback logic."""
    config = ASSISTANTS[assistant_name]
    if attempt == 1:
        return config["primary"]
    elif attempt == 2:
        return config["backup1"]
    elif attempt == 3:
        return config["backup2"]
    else:
        idx = (attempt - 1) % len(RELIABLE_MODELS)
        return RELIABLE_MODELS[idx]

# ------------------------------
# Enhanced Memory System
# ------------------------------
MEMORY_FILE = "memory.json"

class MemoryManager:
    """Enhanced memory management with semantic organization."""
    
    @staticmethod
    def load_memory() -> List[Dict]:
        """Load memory from disk with error handling."""
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "r") as f:
                    memory = json.load(f)
                    return memory if isinstance(memory, list) else []
        except Exception as e:
            st.warning(f"Could not load memory: {e}")
        return []
    
    @staticmethod
    def save_memory(memory: List[Dict]) -> None:
        """Save memory to disk with error handling."""
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(memory, f, indent=2, default=str)
        except Exception as e:
            st.warning(f"Could not save memory: {e}")
    
    @staticmethod
    def prune_memory(memory: List[Dict], retention_days: int, max_items: int) -> List[Dict]:
        """Prune memory based on retention and size limits."""
        if not memory:
            return []
            
        cutoff = datetime.now() - timedelta(days=retention_days)
        pruned = []
        
        for item in memory:
            try:
                ts = datetime.fromisoformat(item.get("ts", ""))
                if ts >= cutoff:
                    pruned.append(item)
            except (ValueError, TypeError):
                continue
        
        return pruned[-max_items:]
    
    @staticmethod
    def detect_financial_facts(text: str) -> List[str]:
        """Enhanced fact detection with better patterns."""
        facts = []
        text_lower = text.lower()
        
        # Income patterns
        income_patterns = [
            r'(?:salary|income|earn|make)\s*(?:of|is)?\s*\$?([\d,\.]+(?:k|K)?)',
            r'\$?([\d,\.]+(?:k|K)?)\s*(?:per year|annually|salary|income)'
        ]
        
        for pattern in income_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                facts.append(f"income:{match.group(1)}")
        
        # Financial goals
        goal_matches = re.findall(r'(?:goal|target|objective)\s*[:\-]?\s*([^\.!?]+)', text, re.IGNORECASE)
        for goal in goal_matches:
            if len(goal.strip()) > 5:
                facts.append(f"goal:{goal.strip()}")
        
        # Risk profile
        risk_keywords = {
            "conservative": ["conservative", "low risk", "safe", "cautious"],
            "moderate": ["moderate", "medium risk", "balanced"],
            "aggressive": ["aggressive", "high risk", "growth", "speculative"]
        }
        
        for profile, keywords in risk_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                facts.append(f"risk_profile:{profile}")
                break
        
        # Time horizons
        time_matches = re.findall(r'(?:time|horizon|years?)\s*(?:of|is)?\s*(\d+\+?\s*(?:years?|months?))', text_lower)
        for time in time_matches:
            facts.append(f"time_horizon:{time.strip()}")
        
        return list(set(facts))
    
    @staticmethod
    def get_memory_context(memory: List[Dict], max_items: int = 15) -> str:
        """Create contextual memory summary for the agent."""
        if not memory:
            return "No relevant memory found."
        
        recent_memory = memory[-max_items:]
        context_lines = ["Relevant user context:"]
        
        for item in recent_memory:
            content = item.get('content', '')
            if content:
                context_lines.append(f"- {content}")
        
        return "\n".join(context_lines)

# ------------------------------
# Enhanced Tools System with Data Visualization
# ------------------------------
class FinanceTools:
    """Enhanced financial tools with better validation and error handling."""
    
    @staticmethod
    def budget_calculation(income: float, bills: float, lifestyle: float) -> Dict[str, Any]:
        """Calculate budget metrics with enhanced analysis."""
        try:
            income_f = float(income)
            bills_f = float(bills) 
            lifestyle_f = float(lifestyle)
            
            savings = income_f - (bills_f + lifestyle_f)
            savings_rate = (savings / income_f) * 100 if income_f > 0 else 0
            essential_ratio = (bills_f / income_f) * 100 if income_f > 0 else 0
            discretionary_ratio = (lifestyle_f / income_f) * 100 if income_f > 0 else 0
            
            return {
                "savings": round(savings, 2),
                "savings_rate_pct": round(savings_rate, 2),
                "essential_ratio_pct": round(essential_ratio, 2),
                "discretionary_ratio_pct": round(discretionary_ratio, 2),
                "analysis": "Healthy" if savings_rate >= 20 else "Needs improvement" if savings_rate >= 10 else "Concerning"
            }
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid input: {e}"}
    
    @staticmethod
    def investment_return(amount: float, rate_pct: float, years: float, contribution: float = 0) -> Dict[str, Any]:
        """Enhanced investment calculator with monthly contributions."""
        try:
            principal = float(amount)
            annual_rate = float(rate_pct) / 100.0
            years_f = float(years)
            monthly_contrib = float(contribution)
            
            # Future value with monthly contributions
            monthly_rate = annual_rate / 12
            months = years_f * 12
            
            if monthly_contrib > 0:
                future_value = principal * ((1 + monthly_rate) ** months)
                future_value += monthly_contrib * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
            else:
                future_value = principal * ((1 + annual_rate) ** years_f)
            
            total_contributions = principal + (monthly_contrib * months)
            total_return = future_value - total_contributions
            return_pct = (total_return / total_contributions) * 100 if total_contributions > 0 else 0
            
            return {
                "future_value": round(future_value, 2),
                "total_contributions": round(total_contributions, 2),
                "total_return": round(total_return, 2),
                "return_percentage": round(return_pct, 2),
                "monthly_contribution": monthly_contrib
            }
        except (ValueError, TypeError) as e:
            return {"error": f"Calculation error: {e}"}
    
    @staticmethod
    def net_worth(assets: Dict[str, float], liabilities: Dict[str, float]) -> Dict[str, Any]:
        """Comprehensive net worth analysis."""
        try:
            total_assets = sum(float(v) for v in assets.values())
            total_liabilities = sum(float(v) for v in liabilities.values())
            net_worth = total_assets - total_liabilities
            debt_to_asset = (total_liabilities / total_assets) * 100 if total_assets > 0 else 0
            
            return {
                "total_assets": round(total_assets, 2),
                "total_liabilities": round(total_liabilities, 2),
                "net_worth": round(net_worth, 2),
                "debt_to_asset_ratio_pct": round(debt_to_asset, 2),
                "financial_health": "Strong" if debt_to_asset < 40 else "Moderate" if debt_to_asset < 70 else "Concerning"
            }
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid asset/liability data: {e}"}
    
    @staticmethod
    def debt_snowball(debts: List[Dict], monthly_payment: float) -> Dict[str, Any]:
        """Enhanced debt snowball calculator."""
        try:
            if not debts:
                return {"error": "No debts provided"}
                
            # Validate debts
            validated_debts = []
            for debt in debts:
                name = debt.get("name", "Unknown")
                balance = float(debt.get("balance", 0))
                rate = float(debt.get("rate", 0))
                min_payment = float(debt.get("min_payment", max(25, balance * 0.02)))
                validated_debts.append({
                    "name": name,
                    "balance": balance,
                    "rate": rate,
                    "min_payment": min_payment
                })
            
            # Sort by balance (snowball method)
            debts_sorted = sorted(validated_debts, key=lambda x: x["balance"])
            
            total_months = 0
            total_interest = 0
            current_balance = sum(d["balance"] for d in debts_sorted)
            plan = []
            
            while current_balance > 0 and total_months < 600:
                available = monthly_payment
                month_interest = 0
                
                # Pay minimums on all debts
                for debt in debts_sorted:
                    if debt["balance"] > 0:
                        interest = debt["balance"] * (debt["rate"] / 100 / 12)
                        month_interest += interest
                        payment = min(debt["min_payment"] + interest, debt["balance"] + interest)
                        actual_payment = min(payment, available)
                        
                        if actual_payment > interest:
                            debt["balance"] -= (actual_payment - interest)
                        
                        available -= actual_payment
                        if available <= 0:
                            break
                
                # Apply extra to smallest debt (snowball)
                if available > 0:
                    for debt in debts_sorted:
                        if debt["balance"] > 0 and available > 0:
                            extra_payment = min(available, debt["balance"])
                            debt["balance"] -= extra_payment
                            available -= extra_payment
                
                total_months += 1
                total_interest += month_interest
                current_balance = sum(d["balance"] for d in debts_sorted)
                
                # Record milestone every 6 months or when debt is paid off
                if total_months % 6 == 0 or current_balance == 0:
                    plan.append({
                        "month": total_months,
                        "remaining_balance": round(current_balance, 2),
                        "total_interest_paid": round(total_interest, 2)
                    })
            
            return {
                "total_months": total_months,
                "total_interest": round(total_interest, 2),
                "total_paid": round(monthly_payment * total_months, 2),
                "plan": plan[:10]
            }
            
        except Exception as e:
            return {"error": f"Debt calculation failed: {e}"}
    
    @staticmethod
    def tax_estimate(annual_salary: float, filing_status: str = "single", state: str = "generic") -> Dict[str, Any]:
        """Enhanced tax estimator with basic brackets."""
        try:
            salary = float(annual_salary)
            
            # Federal tax brackets 2023 (simplified)
            brackets = [
                (11000, 0.10),
                (44725, 0.12),
                (95375, 0.22),
                (182100, 0.24),
                (231250, 0.32),
                (578125, 0.35),
                (float('inf'), 0.37)
            ]
            
            tax = 0.0
            remaining = salary
            prev_bracket = 0
            
            for bracket_max, rate in brackets:
                bracket_size = bracket_max - prev_bracket
                taxable_in_bracket = min(remaining, bracket_size)
                tax += taxable_in_bracket * rate
                remaining -= taxable_in_bracket
                prev_bracket = bracket_max
                if remaining <= 0:
                    break
            
            # Simple FICA taxes
            social_security = min(salary, 160200) * 0.062
            medicare = salary * 0.0145
            
            total_tax = tax + social_security + medicare
            effective_rate = (total_tax / salary) * 100 if salary > 0 else 0
            take_home = salary - total_tax
            
            return {
                "annual_salary": salary,
                "federal_income_tax": round(tax, 2),
                "social_security_tax": round(social_security, 2),
                "medicare_tax": round(medicare, 2),
                "total_tax": round(total_tax, 2),
                "effective_tax_rate_pct": round(effective_rate, 2),
                "annual_take_home": round(take_home, 2),
                "monthly_take_home": round(take_home / 12, 2)
            }
        except (ValueError, TypeError) as e:
            return {"error": f"Tax calculation error: {e}"}
    
    @staticmethod
    def currency_convert(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
        """Currency conversion with fallback."""
        try:
            amount_f = float(amount)
            from_curr = from_currency.upper()
            to_curr = to_currency.upper()
            
            # Try multiple exchange rate APIs
            apis = [
                f"https://api.exchangerate.host/convert?from={from_curr}&to={to_curr}&amount={amount_f}",
                f"https://api.frankfurter.app/latest?from={from_curr}&to={to_curr}&amount={amount_f}"
            ]
            
            for api_url in apis:
                try:
                    response = requests.get(api_url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success", True) and "result" in data:
                            return {
                                "amount": amount_f,
                                "from_currency": from_curr,
                                "to_currency": to_curr,
                                "converted_amount": round(data["result"], 2),
                                "rate": data.get("info", {}).get("rate"),
                                "source": api_url.split('/')[2]
                            }
                except:
                    continue
            
            return {"error": "All conversion APIs failed"}
            
        except (ValueError, TypeError) as e:
            return {"error": f"Conversion error: {e}"}
    
    @staticmethod
    def price_fetch(symbol: str) -> Dict[str, Any]:
        """Enhanced price fetching with multiple fallbacks."""
        try:
            symbol = symbol.strip().upper()
            
            if YFINANCE_AVAILABLE:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    history = ticker.history(period="1d")
                    
                    current_price = (info.get("regularMarketPrice") or 
                                   info.get("currentPrice") or 
                                   info.get("previousClose"))
                    
                    if not current_price and not history.empty:
                        current_price = history['Close'].iloc[-1]
                    
                    if current_price:
                        return {
                            "symbol": symbol,
                            "price": round(current_price, 2),
                            "currency": info.get("currency", "USD"),
                            "source": "yfinance",
                            "name": info.get("shortName", symbol)
                        }
                except Exception as e:
                    st.warning(f"yfinance failed: {e}")
            
            # Fallback to Yahoo API
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    meta = result[0].get("meta", {})
                    price = meta.get("regularMarketPrice") or meta.get("previousClose")
                    if price:
                        return {
                            "symbol": symbol,
                            "price": round(price, 2),
                            "currency": meta.get("currency", "USD"),
                            "source": "yahoo",
                            "name": meta.get("shortName", symbol)
                        }
            
            return {"error": f"Could not fetch price for {symbol}"}
            
        except Exception as e:
            return {"error": f"Price fetch error: {e}"}
    
    @staticmethod
    def invoice_tax_helper(amount: float, tax_pct: float = 10, include_breakdown: bool = True) -> Dict[str, Any]:
        """Enhanced invoice calculator with breakdown."""
        try:
            amount_f = float(amount)
            tax_rate = float(tax_pct) / 100.0
            
            tax_amount = amount_f * tax_rate
            total = amount_f + tax_amount
            
            result = {
                "subtotal": round(amount_f, 2),
                "tax_rate_pct": tax_pct,
                "tax_amount": round(tax_amount, 2),
                "total": round(total, 2)
            }
            
            if include_breakdown:
                result["breakdown"] = {
                    "base_amount": round(amount_f, 2),
                    "tax_calculated": f"{tax_pct}% of ${amount_f:,.2f}",
                    "final_total": round(total, 2)
                }
            
            return result
            
        except (ValueError, TypeError) as e:
            return {"error": f"Invoice calculation error: {e}"}

    # NEW VISUALIZATION TOOLS
    @staticmethod
    def create_budget_chart(income: float, bills: float, lifestyle: float, savings: float) -> Dict[str, Any]:
        """Create a budget breakdown pie chart."""
        try:
            categories = ['Bills', 'Lifestyle', 'Savings']
            values = [bills, lifestyle, savings]
            
            fig = px.pie(
                values=values, 
                names=categories,
                title="Budget Allocation",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            return {
                "chart_type": "pie",
                "title": "Budget Allocation",
                "data": {
                    "categories": categories,
                    "values": values
                },
                "plotly_fig": fig.to_dict()
            }
        except Exception as e:
            return {"error": f"Chart creation failed: {e}"}
    
    @staticmethod
    def create_investment_growth_chart(initial: float, rate: float, years: int, contribution: float = 0) -> Dict[str, Any]:
        """Create investment growth line chart."""
        try:
            years_list = list(range(years + 1))
            growth_data = []
            current_value = initial
            
            for year in years_list:
                growth_data.append(current_value)
                current_value = current_value * (1 + rate/100) + contribution * 12
            
            fig = px.line(
                x=years_list, 
                y=growth_data,
                title="Investment Growth Over Time",
                labels={'x': 'Years', 'y': 'Portfolio Value ($)'}
            )
            fig.update_traces(line=dict(width=3))
            fig.update_layout(
                xaxis_title="Years",
                yaxis_title="Portfolio Value ($)",
                showlegend=False
            )
            
            return {
                "chart_type": "line",
                "title": "Investment Growth",
                "data": {
                    "years": years_list,
                    "values": [round(val, 2) for val in growth_data]
                },
                "plotly_fig": fig.to_dict()
            }
        except Exception as e:
            return {"error": f"Investment chart failed: {e}"}
    
    @staticmethod
    def create_net_worth_chart(assets: Dict[str, float], liabilities: Dict[str, float]) -> Dict[str, Any]:
        """Create net worth composition chart."""
        try:
            # Prepare data for assets pie chart
            asset_items = list(assets.keys())
            asset_values = list(assets.values())
            
            # Prepare data for liabilities pie chart
            liability_items = list(liabilities.keys())
            liability_values = list(liabilities.values())
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "pie"}]],
                subplot_titles=("Assets Composition", "Liabilities Composition")
            )
            
            fig.add_trace(
                go.Pie(labels=asset_items, values=asset_values, name="Assets"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Pie(labels=liability_items, values=liability_values, name="Liabilities"),
                row=1, col=2
            )
            
            fig.update_layout(title_text="Net Worth Composition")
            
            return {
                "chart_type": "subplot_pie",
                "title": "Net Worth Composition",
                "data": {
                    "assets": assets,
                    "liabilities": liabilities
                },
                "plotly_fig": fig.to_dict()
            }
        except Exception as e:
            return {"error": f"Net worth chart failed: {e}"}
    
    @staticmethod
    def create_debt_paydown_chart(debts: List[Dict], monthly_payment: float) -> Dict[str, Any]:
        """Create debt paydown timeline chart."""
        try:
            # Simulate debt paydown
            months = list(range(0, 61, 6))  # 5 years in 6-month increments
            remaining_balances = []
            
            # Simplified simulation
            total_debt = sum(debt.get("balance", 0) for debt in debts)
            for month in months:
                paid = min(total_debt, monthly_payment * month)
                remaining_balances.append(max(0, total_debt - paid))
            
            fig = px.line(
                x=months, 
                y=remaining_balances,
                title="Debt Paydown Timeline",
                labels={'x': 'Months', 'y': 'Remaining Debt ($)'}
            )
            fig.update_traces(line=dict(width=3, color='red'))
            fig.add_hline(y=0, line_dash="dash", line_color="green")
            fig.update_layout(
                xaxis_title="Months",
                yaxis_title="Remaining Debt ($)",
                showlegend=False
            )
            
            return {
                "chart_type": "line",
                "title": "Debt Paydown Timeline",
                "data": {
                    "months": months,
                    "remaining_balance": [round(val, 2) for val in remaining_balances]
                },
                "plotly_fig": fig.to_dict()
            }
        except Exception as e:
            return {"error": f"Debt chart failed: {e}"}
    
    @staticmethod
    def create_stock_history_chart(symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Create stock price history chart."""
        try:
            if not YFINANCE_AVAILABLE:
                return {"error": "yfinance not available"}
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No data available for {symbol}"}
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, 
                y=hist['Close'],
                mode='lines',
                name='Close Price',
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} Stock Price - {period}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_white"
            )
            
            return {
                "chart_type": "stock_history",
                "title": f"{symbol} Price History",
                "data": {
                    "dates": hist.index.strftime('%Y-%m-%d').tolist(),
                    "prices": hist['Close'].round(2).tolist()
                },
                "plotly_fig": fig.to_dict()
            }
        except Exception as e:
            return {"error": f"Stock chart failed: {e}"}
    
    @staticmethod
    def create_tax_comparison_chart(salaries: List[float]) -> Dict[str, Any]:
        """Create tax comparison across different salary levels."""
        try:
            tax_amounts = []
            for salary in salaries:
                tax_result = FinanceTools.tax_estimate(salary)
                if "error" not in tax_result:
                    tax_amounts.append(tax_result["total_tax"])
                else:
                    tax_amounts.append(0)
            
            fig = px.bar(
                x=salaries, 
                y=tax_amounts,
                title="Tax Comparison Across Income Levels",
                labels={'x': 'Annual Salary ($)', 'y': 'Total Tax ($)'}
            )
            fig.update_layout(
                xaxis_title="Annual Salary ($)",
                yaxis_title="Total Tax ($)",
                showlegend=False
            )
            
            return {
                "chart_type": "bar",
                "title": "Tax Comparison",
                "data": {
                    "salaries": salaries,
                    "taxes": [round(tax, 2) for tax in tax_amounts]
                },
                "plotly_fig": fig.to_dict()
            }
        except Exception as e:
            return {"error": f"Tax comparison chart failed: {e}"}

# Tool registry for dynamic execution
TOOL_REGISTRY = {
    "budget_calculation": FinanceTools.budget_calculation,
    "investment_return": FinanceTools.investment_return, 
    "net_worth": FinanceTools.net_worth,
    "debt_snowball": FinanceTools.debt_snowball,
    "tax_estimate": FinanceTools.tax_estimate,
    "currency_convert": FinanceTools.currency_convert,
    "price_fetch": FinanceTools.price_fetch,
    "invoice_tax_helper": FinanceTools.invoice_tax_helper,
    # Visualization tools
    "create_budget_chart": FinanceTools.create_budget_chart,
    "create_investment_growth_chart": FinanceTools.create_investment_growth_chart,
    "create_net_worth_chart": FinanceTools.create_net_worth_chart,
    "create_debt_paydown_chart": FinanceTools.create_debt_paydown_chart,
    "create_stock_history_chart": FinanceTools.create_stock_history_chart,
    "create_tax_comparison_chart": FinanceTools.create_tax_comparison_chart
}

def execute_tool(tool_name: str, args: Dict) -> Dict[str, Any]:
    """Execute tool by name with validated arguments."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}
    
    try:
        return TOOL_REGISTRY[tool_name](**args)
    except TypeError as e:
        return {"error": f"Invalid arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}

def display_plotly_chart(chart_data: Dict[str, Any]) -> None:
    """Display Plotly chart from chart data."""
    try:
        if "plotly_fig" in chart_data:
            fig = go.Figure(chart_data["plotly_fig"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No chart data available to display")
    except Exception as e:
        st.error(f"Failed to display chart: {e}")

# ------------------------------
# Enhanced Agentic System with Visualization Support
# ------------------------------
AGENTIC_SYSTEM_PROMPT = """
You are an autonomous financial AI agent. Your role is to analyze, plan, and execute financial tasks using available tools.

CORE AGENTIC BEHAVIOR:
1. REASONING: Always start with clear reasoning about the user's request
2. PLANNING: Create a step-by-step plan for solving the problem
3. TOOL USAGE: Use available tools for calculations, data fetching, and analysis
4. VISUALIZATION: Create charts and graphs to help users understand complex financial concepts
5. SYNTHESIS: Combine tool results with your knowledge
6. ACTION: Provide specific, actionable recommendations

RESPONSE FORMAT:
You MUST respond with valid JSON only. Choose ONE of these formats:

Tool Execution:
{
  "reasoning": "Brief analysis of the problem",
  "plan": "Step-by-step approach", 
  "action": {
    "name": "tool_name",
    "args": {
      "arg1": value1,
      "arg2": value2
    }
  }
}

Multiple Actions (for complex tasks):
{
  "reasoning": "Brief analysis",
  "plan": "Multi-step approach",
  "actions": [
    {
      "name": "tool_name1",
      "args": { ... }
    },
    {
      "name": "tool_name2", 
      "args": { ... }
    }
  ]
}

Direct Answer:
{
  "reasoning": "Brief analysis", 
  "plan": "Step-by-step approach",
  "answer": "Your complete response to the user"
}

Ask Clarification:
{
  "reasoning": "What information is missing",
  "plan": "How to proceed once we have the information",
  "clarification": "Specific question for the user"
}

AVAILABLE TOOLS:
CALCULATION TOOLS:
- budget_calculation: income, bills, lifestyle expenses
- investment_return: amount, rate_pct, years, [contribution]
- net_worth: assets dict, liabilities dict  
- debt_snowball: debts list, monthly_payment
- tax_estimate: annual_salary, [filing_status], [state]
- currency_convert: amount, from_currency, to_currency
- price_fetch: symbol (stock/crypto)
- invoice_tax_helper: amount, [tax_pct]

VISUALIZATION TOOLS:
- create_budget_chart: income, bills, lifestyle, savings
- create_investment_growth_chart: initial, rate, years, [contribution]
- create_net_worth_chart: assets dict, liabilities dict
- create_debt_paydown_chart: debts list, monthly_payment
- create_stock_history_chart: symbol, [period]
- create_tax_comparison_chart: salaries list

CRITICAL RULES:
- Always include educational disclaimers
- Detect user's language and respond in same language
- Be specific and actionable
- Show your reasoning process
- Use visualization tools to make complex data understandable
- Use tools for all calculations
- For complex analysis, consider using multiple tools in sequence
"""

# ------------------------------
# Streamlit UI - Enhanced with Visualization
# ------------------------------
st.set_page_config(
    page_title="Finance AI Agent (Advanced Agentic + Visualization)", 
    page_icon="💸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Finance AI Agent - Advanced Agentic System with Visualization")
st.caption("Autonomous financial assistant with reasoning, planning, tool execution, and data visualization capabilities. Educational only.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    
    selected_assistant = st.selectbox(
        "Select Assistant:",
        options=list(ASSISTANTS.keys()),
        help="Choose specialized agent for your task"
    )
    
    st.subheader("🧠 Memory Settings")
    retention_days = st.slider("Retention Days", 1, 30, 7)
    max_memory_items = st.slider("Max Memory Items", 10, 500, 200, 10)
    
    st.subheader("🔧 Model Settings") 
    temperature = st.slider("Temperature", 0.1, 1.0, 0.4, 0.1)
    max_tokens = st.slider("Max Tokens", 200, 2000, 800, 100)
    
    st.subheader("📊 Visualization Settings")
    auto_visualize = st.checkbox("Auto-create visualizations", value=True, 
                                help="Automatically create charts for financial data")
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
    
    if st.button("🔄 Clear Memory & Reset"):
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.session_state.clear()
        st.rerun()
    
    st.divider()
    st.caption("🛠️ Available Tools: 8 Calculators + 6 Visualization Tools")
    st.caption("🎯 Agent Features: Reasoning, Planning, Tool Usage, Memory, Visualization, Fallback Models")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager.load_memory()
    st.session_state.memory = MemoryManager.prune_memory(
        st.session_state.memory, retention_days, max_memory_items
    )
if "model_attempts" not in st.session_state:
    st.session_state.model_attempts = {}
if "charts" not in st.session_state:
    st.session_state.charts = []

# Display memory summary
with st.expander("🧠 Agent Memory & Context", expanded=False):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Memory Items:** {len(st.session_state.memory)}")
        if st.session_state.memory:
            st.write("**Recent Context:**")
            for i, item in enumerate(st.session_state.memory[-5:]):
                st.write(f"• {item.get('content', '')}")
    
    with col2:
        if st.button("Clear Session Memory"):
            st.session_state.memory = []
            st.success("Session memory cleared!")
        
        if st.button("Save Memory to Disk"):
            MemoryManager.save_memory(st.session_state.memory)
            st.success("Memory saved!")

# Display visualization gallery if charts exist
if st.session_state.charts:
    with st.expander("📊 Visualization Gallery", expanded=False):
        cols = st.columns(2)
        for i, chart_data in enumerate(st.session_state.charts[-4:]):  # Show last 4 charts
            with cols[i % 2]:
                try:
                    if "plotly_fig" in chart_data:
                        fig = go.Figure(chart_data["plotly_fig"])
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(chart_data.get("title", "Chart"))
                except Exception as e:
                    st.error(f"Could not display chart: {e}")

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant":
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if "timestamp" in message:
                    st.caption(f"🕒 {message['timestamp']}")
            with col2:
                if "model" in message:
                    model_name = message["model"].split("/")[-1]
                    st.caption(f"🧠 {model_name}")
            with col3:
                if message.get("fallback_used"):
                    st.caption("🔄 Fallback Model")

# ------------------------------
# Enhanced Agent Processing Loop with Visualization
# ------------------------------
if prompt := st.chat_input("Ask financial questions... (e.g., budget, investment, tax planning)"):
    # Add user message
    user_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "timestamp": user_ts
    })
    
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
        st.caption(f"Sent: {user_ts}")
    
    # Extract and store financial facts
    financial_facts = MemoryManager.detect_financial_facts(prompt)
    if financial_facts:
        st.session_state.memory = MemoryManager.prune_memory(
            st.session_state.memory + [
                {"ts": datetime.now().isoformat(), "type": "auto", "content": fact}
                for fact in financial_facts
            ],
            retention_days,
            max_memory_items
        )
    
    # Prepare agent context
    assistant_config = ASSISTANTS[selected_assistant]
    combined_prompt = f"{assistant_config['system_prompt']}\n\n{AGENTIC_SYSTEM_PROMPT}"
    
    messages_payload = [
        {"role": "system", "content": combined_prompt},
        {"role": "system", "content": f"Current memory context:\n{MemoryManager.get_memory_context(st.session_state.memory)}"}
    ]
    
    # Add recent conversation (last 6 messages)
    recent_messages = st.session_state.messages[-6:]
    for msg in recent_messages:
        messages_payload.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Agent execution with fallback
    current_attempt = st.session_state.model_attempts.get(selected_assistant, 1)
    max_attempts = 5
    response_text = None
    used_fallback = False
    successful_model = None
    
    for attempt in range(current_attempt, max_attempts + 1):
        model = get_assistant_model(selected_assistant, attempt)
        successful_model = model
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"🤔 Agent reasoning with {model.split('/')[-1]} (attempt {attempt}/{max_attempts})..."):
                response_text = post_openrouter(
                    model, 
                    messages_payload, 
                    temperature=temperature, 
                    max_tokens=max_tokens,
                    timeout=60
                )
        
        if response_text:
            response_text = clean_response(response_text)
            break
        else:
            used_fallback = True
            if attempt < max_attempts:
                st.info(f"🔄 Model failed, trying backup...")
    
    if not response_text:
        st.error("❌ All models failed. Please check API key and try again.")
        st.stop()
    
    # Parse and execute agent response
    agent_response = extract_json_from_text(response_text)
    
    if not agent_response:
        # Fallback: treat as direct response
        bot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": bot_ts,
            "model": successful_model,
            "fallback_used": used_fallback
        })
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(response_text)
            st.caption(f"Responded: {bot_ts}")
    
    elif "clarification" in agent_response:
        # Agent needs clarification
        bot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clarification_msg = f"**Reasoning:** {agent_response.get('reasoning', '')}\n\n**Question:** {agent_response['clarification']}"
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": clarification_msg,
            "timestamp": bot_ts,
            "model": successful_model,
            "fallback_used": used_fallback
        })
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write("🤔 **Agent Reasoning:**")
            st.write(agent_response.get('reasoning', ''))
            st.write("💡 **Plan:**")
            st.write(agent_response.get('plan', ''))
            st.write("❓ **Clarification Needed:**")
            st.info(agent_response['clarification'])
            st.caption(f"Reasoned: {bot_ts}")
    
    elif "action" in agent_response or "actions" in agent_response:
        # Tool execution path (single or multiple actions)
        actions = agent_response.get("actions", [agent_response["action"]]) if "actions" in agent_response else [agent_response["action"]]
        
        # Store agent's reasoning
        reasoning_msg = f"**Reasoning:** {agent_response.get('reasoning', '')}\n\n**Plan:** {agent_response.get('plan', '')}\n\n**Actions:** Executing {len(actions)} tool(s)"
        st.session_state.messages.append({
            "role": "assistant",
            "content": reasoning_msg,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": successful_model,
            "fallback_used": used_fallback
        })
        
        # Show reasoning
        with st.chat_message("assistant", avatar="🤖"):
            st.write("🧠 **Agent Reasoning:**")
            st.write(agent_response.get('reasoning', ''))
            st.write("📋 **Execution Plan:**")
            st.write(agent_response.get('plan', ''))
            st.write("🛠️ **Tool Execution:**")
        
        all_tool_results = []
        
        for i, action in enumerate(actions):
            tool_name = action["name"]
            tool_args = action.get("args", {})
            
            with st.chat_message("assistant", avatar="🤖"):
                st.code(f"Tool {i+1}: {tool_name}\nArgs: {json.dumps(tool_args, indent=2)}")
            
            # Execute tool
            with st.spinner(f"Executing {tool_name}..."):
                tool_result = execute_tool(tool_name, tool_args)
                all_tool_results.append({"tool": tool_name, "result": tool_result})
            
            # Show tool results
            with st.chat_message("assistant", avatar="🤖"):
                st.write(f"📊 **Tool {i+1} Results:**")
                
                # Check if this is a visualization tool
                if tool_name.startswith("create_") and "plotly_fig" in tool_result:
                    st.write(f"**Chart:** {tool_result.get('title', 'Visualization')}")
                    display_plotly_chart(tool_result)
                    # Store chart for gallery
                    st.session_state.charts.append(tool_result)
                else:
                    st.json(tool_result)
        
        # Generate final answer with all tool results
        followup_messages = messages_payload + [
            {"role": "assistant", "content": response_text},
            {"role": "tool", "content": json.dumps({
                "tools_executed": len(actions),
                "results": all_tool_results
            })},
            {"role": "user", "content": "Using the tool results above, provide a comprehensive final answer with specific recommendations, educational disclaimers, and insights from the visualizations if any were created."}
        ]
        
        with st.spinner("🔄 Synthesizing final answer..."):
            final_response = post_openrouter(
                successful_model,
                followup_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        final_text = clean_response(final_response) if final_response else "Tools executed successfully but could not generate final answer."
        
        # Store and display final answer
        bot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_text,
            "timestamp": bot_ts,
            "model": successful_model,
            "fallback_used": used_fallback,
            "tools_used": [action["name"] for action in actions]
        })
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write("💡 **Final Recommendation:**")
            st.write(final_text)
            st.caption(f"Completed: {bot_ts}")
    
    elif "answer" in agent_response:
        # Direct answer path
        bot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        answer_msg = f"**Reasoning:** {agent_response.get('reasoning', '')}\n\n**Plan:** {agent_response.get('plan', '')}\n\n**Answer:** {agent_response['answer']}"
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_msg,
            "timestamp": bot_ts,
            "model": successful_model,
            "fallback_used": used_fallback
        })
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write("🧠 **Agent Reasoning:**")
            st.write(agent_response.get('reasoning', ''))
            st.write("📋 **Execution Plan:**")
            st.write(agent_response.get('plan', ''))
            st.write("💡 **Final Answer:**")
            st.write(agent_response['answer'])
            st.caption(f"Completed: {bot_ts}")
    
    # Update model attempts for next interaction
    st.session_state.model_attempts[selected_assistant] = current_attempt
    
    # Save memory
    MemoryManager.save_memory(st.session_state.memory)

# Footer
st.divider()
st.caption("💡 **Agent Capabilities:** Reasoning • Planning • Tool Execution • Data Visualization • Memory • Multi-model Fallback")
st.caption("📊 **Visualization Tools:** Budget Charts • Investment Growth • Net Worth • Debt Paydown • Stock History • Tax Comparisons")
st.caption("🔐 **Disclaimer:** Educational purposes only. Not financial advice. Always consult qualified professionals.")
st.caption("⚡ **Tips:** Be specific with numbers and goals. Use specialized agents for complex tasks. Ask for visualizations!")

# --- Assumed main logic block follows, requires a context like a function or 'if' block ---
selected_name = selected_assistant_name
current_attempt = st.session_state.model_attempts.get(selected_name, 1)
max_attempts = 6
response_text = None
used_fallback = False
successful_attempt = current_attempt

for attempt in range(current_attempt, max_attempts + 1):
    model_to_try = get_assistant_model(selected_name, attempt)
    with st.chat_message("assistant", avatar="🤖"):
        status = f"Thinking with model {model_to_try} (attempt {attempt}/{max_attempts})..."
        with st.spinner(status):
            # This line and the rest of the block are assumed to be inside a function/conditional statement
            # You must ensure 'post_openrouter', 'get_assistant_model', etc., are defined.
            raw = post_openrouter(model_to_try, messages_payload, temperature=st.session_state.temperature, max_tokens=st.session_state.max_tokens)
    
    if raw:
        response_text = clean_response(raw)
        successful_attempt = attempt
        break
    else:
        # try next
        used_fallback = True
        st.info("Attempt failed — trying next model...")

if not response_text:
    st.error("All models failed. Please check your API key or network.")
else:
    # First response should be JSON according to AGENT_SYSTEM_PROMPT; try to parse it
    parsed = extract_json_from_text(response_text)
    tool_result = None
    tool_called = None

    if parsed and "action" in parsed:
        action = parsed.get("action", {})
        tool_name = action.get("name")
        args = action.get("args", {})
        tool_called = tool_name
        
        # Execute the requested tool (map names)
        if tool_name == "budget_calculation":
            tool_result = tool_budget_calculation(args.get("income"), args.get("bills"), args.get("lifestyle"))
        elif tool_name == "investment_return":
            tool_result = tool_investment_return(args.get("amount"), args.get("rate_pct"), args.get("years"))
        elif tool_name == "net_worth":
            tool_result = tool_net_worth(args.get("assets", {}), args.get("liabilities", {}))
        elif tool_name == "debt_snowball":
            tool_result = tool_debt_snowball(args.get("debts", []))
        elif tool_name == "tax_estimate":
            tool_result = tool_tax_estimate(args.get("annual_salary"), args.get("country","generic"))
        elif tool_name == "currency_convert":
            tool_result = tool_currency_convert(args.get("amount"), args.get("from_currency"), args.get("to_currency"))
        elif tool_name == "price_fetch":
            tool_result = tool_price_fetch(args.get("symbol"))
        elif tool_name == "invoice_tax_helper":
            # Note: The original code had a typo here, changed "tax_pct",10) to "tax_pct", 10)
            tool_result = tool_invoice_tax_helper(args.get("amount"), args.get("tax_pct", 10))
        else:
            tool_result = {"error": "Unknown tool requested."}
            
# The tool execution logic ends here.

            # Append the assistant's plan and action as message, and append a tool role with results
            st.session_state.messages.append({"role":"assistant","content":response_text,"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": get_assistant_model(selected_name, successful_attempt), "fallback_used": used_fallback})
            # show assistant plan
            with st.chat_message("assistant", avatar="🤖"):
                st.write(f"Plan: {parsed.get('plan')}")
                st.caption("Agent decided to run an internal tool and returned results below...")

            # show tool result to user immediately
            with st.chat_message("assistant", avatar="🤖"):
                st.write("🔧 Tool result:")
                st.write(tool_result)

            # now call the model again with tool result appended so it can generate final answer
            followup_msgs = [{"role":"system", "content": combined_system_prompt}]
            # memory summary again
            if st.session_state.memory:
                mem_summary = "\n".join([f"- {m['content']}" for m in st.session_state.memory[-10:]])
                followup_msgs.append({"role":"system","content":f"User memory summary (short):\n{mem_summary}"})
            # conversation history
            for m in st.session_state.messages[-10:]:
                followup_msgs.append({"role": m["role"], "content": m["content"]})
            # add tool result as a 'tool' role so model sees it
            followup_msgs.append({"role":"tool", "content": json.dumps({"tool": tool_name, "result": tool_result})})
            # final ask: produce a friendly answer using the tool result
            followup_msgs.append({"role":"user", "content":"Using the tool result above, produce a concise friendly answer for the user (include disclaimer)."})
            final_response = post_openrouter(get_assistant_model(selected_name, successful_attempt), followup_msgs, temperature=st.session_state.temperature, max_tokens=st.session_state.max_tokens)
            final_text = clean_response(final_response) if final_response else "Tool executed but assistant could not produce a final answer."

            # store memory if the user gave important facts or agent suggests storing
            mem_facts = detect_memoryworthy_facts(prompt + " " + json.dumps(tool_result))
            if mem_facts:
                store_memory_items(mem_facts)

            # append final assistant message to session chat
            bot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg_data = {"role":"assistant","content": final_text, "timestamp": bot_ts, "model": get_assistant_model(selected_name, successful_attempt)}
            if used_fallback:
                msg_data["fallback_used"] = True
            st.session_state.messages.append(msg_data)

            # display final assistant message
            with st.chat_message("assistant", avatar="🤖"):
                st.write(final_text)
                st.caption(f"Responded at: {bot_ts}")
                model_name = get_assistant_model(selected_name, successful_attempt).split('/')[1] if "/" in get_assistant_model(selected_name, successful_attempt) else get_assistant_model(selected_name, successful_attempt)
                if used_fallback:
                    st.caption(f"🔄 Model: {model_name} (Fallback)")
                else:
                    st.caption(f"Model: {model_name}")

# Assuming this code is inside a larger 'if tool_result is not None:' block, 
# or follows the tool execution logic.

# The initial indentation of '            ' is removed to fit the likely context 
# where 'tool_result' or 'parsed' was checked.

if tool_result is not None:
    # This block executes if a tool was called and a result was obtained.
    # Note: 'final_text' must be defined and constructed based on the tool_result 
    # before this part of the code is reached.
    
    # display final assistant message
    with st.chat_message("assistant", avatar="🤖"):
        st.write(final_text)
        st.caption(f"Responded at: {bot_ts}")
        
        # Determine model name for display
        model_full_name = get_assistant_model(selected_name, successful_attempt)
        model_name = model_full_name.split('/')[-1] # Cleaner way to get the base name
        
        if used_fallback:
            st.caption(f"🔄 **Model:** {model_name} (Fallback)")
        else:
            st.caption(f"**Model:** {model_name}")

# Append the assistant's plan and action as message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        "model": get_assistant_model(selected_name, successful_attempt), 
        "fallback_used": used_fallback
    })
    
    # show assistant plan
    with st.chat_message("assistant", avatar="🤖"):
        st.write(f"Plan: {parsed.get('plan')}")
        st.caption("Agent decided to run an internal tool and returned results below...")

    # show tool result to user immediately
    with st.chat_message("assistant", avatar="🤖"):
        st.write("🔧 Tool result:")
        st.code(tool_result, language='json') # Use st.code for structured output

    # --- Tool Follow-up Call (to generate final answer) ---
    
    followup_msgs = [{"role":"system", "content": combined_system_prompt}]
    
    # memory summary again
    if st.session_state.memory:
        mem_summary = "\n".join([f"- {m['content']}" for m in st.session_state.memory[-10:]])
        followup_msgs.append({"role":"system","content":f"User memory summary (short):\n{mem_summary}"})
        
    # conversation history
    for m in st.session_state.messages[-10:]:
        followup_msgs.append({"role": m["role"], "content": m["content"]})
        
    # add tool result as a 'tool' role so model sees it
    followup_msgs.append({"role":"tool", "content": json.dumps({"tool": tool_name, "result": tool_result})})
    
    # final ask: produce a friendly answer using the tool result
    followup_msgs.append({"role":"user", "content":"Using the tool result above, produce a concise friendly answer for the user (include disclaimer)."})
    
    final_response = post_openrouter(
        get_assistant_model(selected_name, successful_attempt), 
        followup_msgs, 
        temperature=st.session_state.temperature, 
        max_tokens=st.session_state.max_tokens
    )
    final_text = clean_response(final_response) if final_response else "Tool executed but assistant could not produce a final answer."

    # store memory if the user gave important facts or agent suggests storing
    mem_facts = detect_memoryworthy_facts(prompt + " " + json.dumps(tool_result))
    if mem_facts:
        store_memory_items(mem_facts)

    # append final assistant message to session chat
    bot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg_data = {
        "role": "assistant",
        "content": final_text, 
        "timestamp": bot_ts, 
        "model": get_assistant_model(selected_name, successful_attempt)
    }
    if used_fallback:
        msg_data["fallback_used"] = True
    st.session_state.messages.append(msg_data)

    # display final assistant message
    with st.chat_message("assistant", avatar="🤖"):
        st.write(final_text)
        st.caption(f"Responded at: {bot_ts}")
        
        model_full_name = get_assistant_model(selected_name, successful_attempt)
        model_name = model_full_name.split('/')[-1] # Cleaner way to get the base name
        
        if used_fallback:
            st.caption(f"🔄 **Model:** {model_name} (Fallback)")
        else:
            st.caption(f"**Model:** {model_name}")

else:
    # --- Logic for Direct Response (No tool called) ---
    
    # model did not request a tool: parse answer and show directly
    bot_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ensure we append assistant full message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "timestamp": bot_ts,
        "model": get_assistant_model(selected_name, successful_attempt)
    })
    
    with st.chat_message("assistant", avatar="🤖"):
        st.write(response_text)
        st.caption(f"Responded at: {bot_ts}")
        
        model_full_name = get_assistant_model(selected_name, successful_attempt)
        model_name = model_full_name.split('/')[-1]
        
        if used_fallback:
            st.caption(f"🔄 **Model:** {model_name} (Fallback)")
        else:
            st.caption(f"**Model:** {model_name}")

# Footer
st.divider()
st.caption("💡 Tip: Ask focused finance questions and provide numbers. Memory retention is short by default; change it in the sidebar.")
st.caption("🔐 Educational only — not professional financial advice.")
