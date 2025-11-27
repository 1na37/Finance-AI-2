import streamlit as st
import requests
import json
from datetime import datetime

# ------------------------------
# Finance-focused Multi-Assistant
# Single-file Streamlit app
# ------------------------------

# Validate API key before running
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("❌ OpenRouter API key not found. Please add it to your secrets.toml file.")
    st.stop()


def clean_response(content: str) -> str:
    """Clean up AI-generated text from common markup artifacts."""
    if content:
        content = content.replace('```', '').replace('**', '').replace('*', '').strip()
        content = content.replace('<s>', '').replace('</s>', '').strip()
        # Remove empty lines and trim whitespace
        content = '\n'.join([line.rstrip() for line in content.split('\n') if line.strip()])
    return content


def get_ai_response(messages_payload, model, temperature=0.7, max_tokens=500):
    api_key = st.secrets["OPENROUTER_API_KEY"]
    try:
        # Rate limiting: simple per-session guard
        if "last_request_time" in st.session_state:
            time_diff = datetime.now() - st.session_state.last_request_time
            if time_diff.total_seconds() < 0.8:  # limit to ~1 request/sec
                st.warning("⚠️ Please wait a moment before sending another message")
                return None

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            data=json.dumps({
                "model": model,
                "messages": messages_payload,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }),
            timeout=30
        )

        # update last request timestamp
        st.session_state.last_request_time = datetime.now()

        if response.status_code != 200:
            # show short error but don't leak long response bodies
            st.error(f"API Error {model}: {response.status_code} - {response.text[:200]}")
            return None

        data = response.json()
        # Defensive access pattern
        answer = None
        if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            # compatibility with different provider shapes
            if "message" in choice and isinstance(choice["message"], dict):
                answer = choice["message"].get("content")
            elif "text" in choice:
                answer = choice.get("text")

        return clean_response(answer) if answer else None

    except requests.exceptions.Timeout:
        st.error(f"Request timeout for {model}. Please try again.")
        return None
    except Exception as e:
        st.error(f"Request failed for {model}: {str(e)}")
        return None


def run_chain(prompt, base_messages, current_model):
    """Multi-agent financial analysis chain"""
    results = {}
    
    # Show chain progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Agent 1 – Extractor
    status_text.text("🔍 Extracting financial numbers...")
    progress_bar.progress(25)
    messages1 = base_messages + [
        {"role": "system", "content": 
         "Extract all financial numbers and relevant variables from the user's message. "
         "Return in JSON with keys: income, expenses, debt, savings, goals, timeframe, unknown."},
        {"role": "user", "content": prompt}
    ]
    results["extracted"] = get_ai_response(messages1, current_model)

    # Agent 2 – Analyst
    status_text.text("📊 Analyzing financial situation...")
    progress_bar.progress(50)
    messages2 = base_messages + [
        {"role": "system", "content":
         "Using extracted finance data, analyze the user's financial situation. "
         "Return bullet list: strengths, risks, opportunities, warnings. Keep concise."},
        {"role": "assistant", "content": results["extracted"]},
    ]
    results["analysis"] = get_ai_response(messages2, current_model)

    # Agent 3 – Recommendation Maker
    status_text.text("💡 Creating financial plan...")
    progress_bar.progress(75)
    messages3 = base_messages + [
        {"role": "system", "content":
         "Using the risk & opportunity analysis, give a final financial plan: monthly budget, "
         "short-term steps, long-term strategy, risk warnings. Make it practical and friendly."},
        {"role": "assistant", "content": results["analysis"]},
    ]
    results["final"] = get_ai_response(messages3, current_model)

    # Complete progress
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    return results


# Page configuration
st.set_page_config(page_title="Finance AI Assistant", page_icon="💸", layout="wide")
st.title("💸 Finance AI Assistant — Multi-Agent")
st.caption("Automated financial analysis: extraction → analysis → personalized recommendations")

# ------------------------------
# Finance-focused assistant configurations
# ------------------------------
assistants = {
    "💼 Personal Finance Advisor": {
        "primary": "qwen/qwen3-235b-a22b:free",
        "backup1": "deepseek/deepseek-chat-v3.1:free",
        "backup2": "mistralai/mistral-small-3.2-24b-instruct:free",
        "system_prompt": (
            "You are a helpful Personal Finance Advisor. Provide clear, practical guidance on budgeting, saving, debt management, "
            "emergency funds, and simple investment basics. When giving examples use conservative, realistic numbers. "
            "Always include a brief, explicit disclaimer: 'I am not a licensed financial advisor; this is educational information.'"
        ),
        "reason": "Qwen 3 offers strong multilingual support and structured outputs for budgets and personal finance"
    },
    "📈 Investment Analyst": {
        "primary": "deepseek/deepseek-r1-0528:free",
        "backup1": "openai/gpt-oss-120b:free",
        "backup2": "qwen/qwen3-coder:free",
        "system_prompt": (
            "You are an Investment Analyst assistant. Provide objective analysis of asset classes, valuation concepts (P/E, yield, ROE), "
            "portfolio construction basics, risk management, and scenario analysis. Use clear assumptions and show math steps when appropriate. "
            "Include the educational disclaimer: 'Not professional financial advice.'"
        ),
        "reason": "DeepSeek / GPT-OSS are strong at multi-step reasoning and numeric work"
    },
    "🔗 Multi-Agent Financial Chain": {
        "primary": "qwen/qwen3-235b-a22b:free",
        "backup1": "deepseek/deepseek-chat-v3.1:free",
        "backup2": "google/gemini-2.0-flash-exp:free",
        "system_prompt": (
            "You are part of a multi-agent financial analysis system. Provide clear, structured responses that can be used by subsequent agents. "
            "Always include educational disclaimers about not being licensed financial advice."
        ),
        "reason": "Multi-agent chain for automated financial analysis"
    }
}

# Fallback reliable models ranking
RELIABLE_MODELS = [
    "qwen/qwen3-235b-a22b:free",
    "deepseek/deepseek-chat-v3.1:free",
    "google/gemini-2.0-flash-exp:free",
    "x-ai/grok-4-fast:free",
    "mistralai/mistral-small-3.2-24b-instruct:free"
]


def get_assistant_model(assistant_name, attempt=1):
    config = assistants[assistant_name]
    if attempt == 1:
        return config["primary"]
    elif attempt == 2:
        return config["backup1"]
    elif attempt == 3:
        return config["backup2"]
    elif 4 <= attempt <= 8:
        idx = attempt - 4
        return RELIABLE_MODELS[idx % len(RELIABLE_MODELS)]
    else:
        return RELIABLE_MODELS[0]

# ------------------------------
# Session state initialization
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_attempts" not in st.session_state:
    st.session_state.model_attempts = {}
if "current_assistant" not in st.session_state:
    st.session_state.current_assistant = "🔗 Multi-Agent Financial Chain"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.4
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 600
if "used_fallback" not in st.session_state:
    st.session_state.used_fallback = False
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = datetime.now()
if "language" not in st.session_state:
    st.session_state.language = "Auto (match input)"

# ------------------------------
# Sidebar UI (Simplified)
# ------------------------------
with st.sidebar:
    st.header("⚙️ Konfigurasi Asisten Keuangan")

    # API status
    if "OPENROUTER_API_KEY" in st.secrets:
        st.success("✅ API Key: Configured")
    else:
        st.error("❌ API Key: Missing")

    selected_assistant_name = st.selectbox("Pilih Asisten:", options=list(assistants.keys()))

    if selected_assistant_name != st.session_state.current_assistant:
        st.session_state.current_assistant = selected_assistant_name
        if selected_assistant_name not in st.session_state.model_attempts:
            st.session_state.model_attempts[selected_assistant_name] = 1

    current_attempt = st.session_state.model_attempts.get(selected_assistant_name, 1)
    current_model = get_assistant_model(selected_assistant_name, current_attempt)

    # Language options
    st.session_state.language = st.selectbox("Language / Bahasa:", options=[
        "Auto (match input)", "English", "Bahasa Indonesia"
    ], index=0)

    # Show model status
    if current_attempt <= 3:
        st.success(f"Model: {current_model.split('/')[1]}")
        st.caption("✅ Using assistant-specific model")
    elif current_attempt <= 8:
        st.warning(f"Model: {current_model.split('/')[1]}")
        st.caption("🔄 Using universal reliable model")
    else:
        st.error(f"Model: {current_model.split('/')[1]}")
        st.caption("🚨 Using ultimate fallback model")

    st.caption(f"💡 {assistants[selected_assistant_name]['reason']}")

    # Temperature and token settings
    st.session_state.temperature = st.slider(
        "Temperature:", 0.0, 1.0, value=st.session_state.temperature, step=0.05,
        help="Lower = more deterministic. Higher = more creative."
    )

    st.session_state.max_tokens = st.slider(
        "Max response tokens:", 100, 2000, value=st.session_state.max_tokens, step=50,
        help="Limit the length of AI responses"
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"💬 Pesan dalam chat: {len(st.session_state.messages)}")

# ------------------------------
# Display chat messages
# ------------------------------
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        if "timestamp" in message:
            st.caption(message["timestamp"])
        if message["role"] == "assistant" and "model" in message:
            model_name = message["model"].split('/')[1]
            if message.get("fallback_used"):
                st.caption(f"🔄 Model: {model_name} (Fallback)")
            else:
                st.caption(f"Model: {model_name}")

# ------------------------------
# Chat input with Agent Chain
# ------------------------------
if prompt := st.chat_input("Tanyakan hal finansial Anda... (mis. Gaji 6 juta, pengeluaran 3.5 juta, nabung susah)"):
    # Add user message
    user_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": user_timestamp})

    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
        st.caption(f"Sent at: {user_timestamp}")

    # Build system prompt with language & assistant role
    assistant_cfg = assistants[st.session_state.current_assistant]
    lang_instruction = ""
    if st.session_state.language != "Auto (match input)":
        lang_instruction = f"Respond in {st.session_state.language}."
    else:
        # prefer matching user's input language; assistant should detect and match
        lang_instruction = "Detect the user's language and respond in the same language."

    # Combine system prompt with language instruction and finance disclaimer
    combined_system_prompt = assistant_cfg["system_prompt"] + " " + lang_instruction

    # Limit message history to last 10 messages for context
    recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages

    messages_with_system = [{"role": "system", "content": combined_system_prompt}]
    for msg in recent_messages:
        if msg["role"] in ["user", "assistant"]:
            messages_with_system.append({"role": msg["role"], "content": msg["content"]})

    # ULTIMATE fallback attempts across multiple models
    selected_name = st.session_state.current_assistant
    current_attempt = st.session_state.model_attempts.get(selected_name, 1)
    max_attempts = 8
    response = None
    used_fallback = False
    successful_attempt = current_attempt

    # Check if we should use agent chain (only for multi-agent assistant)
    use_agent_chain = selected_name == "🔗 Multi-Agent Financial Chain"

    for attempt in range(current_attempt, max_attempts + 1):
        try_model = get_assistant_model(selected_name, attempt)
        with st.chat_message("assistant", avatar="🤖"):
            if attempt <= 3:
                status_text = f"Trying assistant model {attempt}/3: {try_model.split('/')[1]}..."
            else:
                status_text = f"Trying universal fallback {attempt-3}/5: {try_model.split('/')[1]}..."
                used_fallback = True

            with st.spinner(status_text):
                if use_agent_chain:
                    # Use the multi-agent chain
                    chain_output = run_chain(prompt, messages_with_system, try_model)
                    response = chain_output["final"]
                else:
                    # Use regular single-response mode
                    response = get_ai_response(
                        messages_with_system,
                        try_model,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens
                    )

        if response:
            successful_attempt = attempt
            break
        else:
            if attempt == 3:
                st.warning("Assistant-specific models failed, switching to reliable fallback models...")
            else:
                st.info("Attempt failed — trying next model...")

    # If still no response, try ultimate fallback explicitly
    if not response:
        st.error("All primary attempts failed. Trying ultimate fallback...")
        ultimate = RELIABLE_MODELS[0]
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"Emergency fallback: {ultimate.split('/')[1]}..."):
                if use_agent_chain:
                    chain_output = run_chain(prompt, messages_with_system, ultimate)
                    response = chain_output["final"]
                else:
                    response = get_ai_response(messages_with_system, ultimate,
                                               temperature=st.session_state.temperature,
                                               max_tokens=st.session_state.max_tokens)
        if response:
            successful_attempt = max_attempts + 1
            used_fallback = True

    # Append response to session state
    if response:
        st.session_state.model_attempts[selected_name] = successful_attempt
        st.session_state.used_fallback = used_fallback

        bot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_model = get_assistant_model(selected_name, successful_attempt)

        message_data = {
            "role": "assistant",
            "content": response,
            "timestamp": bot_timestamp,
            "model": final_model
        }
        if used_fallback:
            message_data["fallback_used"] = True

        st.session_state.messages.append(message_data)

        # Display the AI's response
        st.write(response)
        st.caption(f"Responded at: {bot_timestamp}")
        model_name = final_model.split('/')[1]
        if used_fallback:
            st.caption(f"🔄 Model: {model_name} (Fallback)")
        else:
            st.caption(f"Model: {model_name}")

    else:
        st.error("❌ All models failed to produce a response. Check your API key, network, or try another model.")

# Footer: tips & disclaimers
st.divider()
st.caption("💡 Tip: For automated financial analysis, use 'Multi-Agent Financial Chain'. It will extract numbers, analyze your situation, and create personalized recommendations automatically.")
st.caption("🔐 This assistant provides educational information and should not be used as professional financial advice.")
