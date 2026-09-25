
import os
import certifi
import requests
import streamlit as st #weather

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langsmith import Client


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Search + Weather AI Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        /* Main app */
        .main {
            background-color: #0e1117;
        }

        /* Header */
        .hero {
            padding: 1.5rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.2rem;
            background: linear-gradient(135deg, #172033 0%, #111827 100%);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            color: #aab4c3;
            font-size: 1rem;
        }

        /* Tool cards */
        .tool-card {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            background: #161b22;
            border: 1px solid #30363d;
        }

        .tool-name {
            font-weight: 700;
            color: #8ab4f8;
        }

        /* Source cards */
        .source-card {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin: 0.6rem 0;
            background: #161b22;
            border: 1px solid #30363d;
        }

        .source-card a {
            text-decoration: none;
            font-weight: 600;
        }

        /* Status */
        .status-ok {
            color: #3fb950;
            font-weight: 600;
        }

        .status-bad {
            color: #f85149;
            font-weight: 600;
        }

        /* Hide Streamlit footer */
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


# ============================================================
# CUSTOM WEATHER TOOL
# ============================================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather data for a given city.
    """
    if not WEATHER_API_KEY:
        return "Weather API key is not configured."

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHER_API_KEY}&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Weather service request failed: {e}"

    if "current" not in data:
        error = data.get("error", {}).get("info", "Unknown weather API error")
        return f"Could not find weather data for {city}. {error}"

    current = data["current"]
    description = current.get("weather_descriptions", ["Unknown"])[0]

    return (
        f"City : {city}\n"
        f"Temperature : {current.get('temperature', 'N/A')}°C\n"
        f"Weather : {description}\n"
        f"Humidity : {current.get('humidity', 'N/A')}%"
    )


# ============================================================
# BUILD AGENT
# Cached so Streamlit does not rebuild everything on every
# interaction.
# ============================================================

@st.cache_resource(show_spinner=False)
def build_agent():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing from your .env file.")

    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is missing from your .env file.")

    if not WEATHER_API_KEY:
        raise ValueError("WEATHER_API_KEY is missing from your .env file.")

    search_tool = TavilySearchResults(max_results=2)

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.6-flash",
        temperature=0,
        api_key=GEMINI_API_KEY,
    )

    client = Client()

    prompt = client.pull_prompt(
        "hwchase17/react",
        dangerously_pull_public_prompt=True,
    )

    tools = [search_tool, get_weather_data]

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return agent_executor


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_steps" not in st.session_state:
    st.session_state.last_steps = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ Agent Control")

    st.markdown("### Available tools")

    st.success("🔎 Tavily Web Search")
    st.success("🌤️ WeatherStack Weather")

    st.divider()

    st.markdown("### API status")

    if GEMINI_API_KEY:
        st.markdown('<span class="status-ok">● Gemini configured</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-bad">● Gemini key missing</span>',
                    unsafe_allow_html=True)

    if TAVILY_API_KEY:
        st.markdown('<span class="status-ok">● Tavily configured</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-bad">● Tavily key missing</span>',
                    unsafe_allow_html=True)

    if WEATHER_API_KEY:
        st.markdown('<span class="status-ok">● Weather API configured</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-bad">● Weather key missing</span>',
                    unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_steps = []
        st.rerun()

    st.divider()

    st.caption("Search + Weather AI Agent")
    st.caption("Built with LangChain • Gemini • Tavily • WeatherStack")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌍 Search + Weather AI Agent</h1>
        <p>
            Ask questions, search the web, or get current weather information.
            The agent automatically chooses the appropriate tool.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLE PROMPTS
# ============================================================

st.markdown("#### Try an example")

example_cols = st.columns(3)

examples = [
    "🌤️ What is the current weather in Tokyo?",
    "🔎 Who is the current CEO of NVIDIA?",
    "🌍 What is the capital of Japan?",
]

for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.pending_prompt = example
        st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# INPUT
# ============================================================

pending_prompt = st.session_state.pop("pending_prompt", None)

user_input = st.chat_input(
    "Ask me about the web or weather..."
)

if pending_prompt:
    user_input = pending_prompt


# ============================================================
# AGENT EXECUTION
# ============================================================

if user_input:
    # Display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            agent_executor = build_agent()

            with st.status("🤖 Agent is working...", expanded=True) as status:
                st.write("Analyzing your request...")

                response = agent_executor.invoke(
                    {
                        "input": user_input
                    }
                )

                intermediate_steps = response.get(
                    "intermediate_steps", []
                )

                st.session_state.last_steps = intermediate_steps

                if intermediate_steps:
                    st.write(
                        f"Used {len(intermediate_steps)} tool step(s)."
                    )

                status.update(
                    label="✅ Completed",
                    state="complete",
                    expanded=False,
                )

            final_answer = response.get(
                "output",
                "I couldn't generate a final answer."
            )

            st.markdown(final_answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                }
            )

        except Exception as e:
            st.error("Something went wrong.")

            with st.expander("Show technical error"):
                st.code(str(e))

            st.info(
                "Check your .env keys, Gemini quota, Tavily configuration, "
                "and WeatherStack configuration."
            )


# ============================================================
# TOOL ACTIVITY
# ============================================================

if st.session_state.last_steps:
    st.divider()

    with st.expander("🔧 View agent tool activity"):
        for index, step in enumerate(
            st.session_state.last_steps, start=1
        ):
            try:
                action, observation = step

                st.markdown(
                    f'<div class="tool-card">'
                    f'<div class="tool-name">Step {index}: '
                    f'{action.tool}</div>'
                    f'<div><b>Input:</b> '
                    f'{action.tool_input}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Keep raw observation readable without making the
                # main chat noisy.
                st.code(str(observation), language="text")

            except Exception:
                st.code(str(step), language="text")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "AI responses may contain outdated or incorrect information. "
    "Verify important information with reliable sources."
)
