"""
Streamlit Chat UI for NL2SQL Demo.

INTERVIEW NOTE: Shows generated SQL for transparency.
Run with: streamlit run databricks_prototype/app/main.py
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.sql_agent import DatabricksNL2SQLAgent

st.set_page_config(page_title="Ryder AI Assistant", page_icon="🚚", layout="wide")

st.title("🚚 Ryder AI Assistant")
st.markdown("*NL2SQL on Delta Tables - Databricks Mosaic AI*")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    catalog = st.text_input("Catalog", value="ryder_chatbot")
    schema = st.selectbox("Schema", options=["agent_accessible"])

    st.divider()
    st.markdown("### 📝 Sample Questions")
    st.markdown("""
    - "Which vendors have reliability > 95%?"
    - "Show routes from Phoenix hub"
    - "Count questions by topic"
    - "List all vendors sorted by performance"
    """)

    st.divider()
    st.markdown("### 🔒 Security")
    st.markdown("""
    **Defense-in-depth:**
    1. Unity Catalog RBAC
    2. Hardened system prompt
    3. SQL validator (AST parsing)
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your data..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # Cache the agent for performance
            @st.cache_resource
            def get_agent(cat: str, sch: str) -> DatabricksNL2SQLAgent:
                """Create and cache the NL2SQL agent."""
                return DatabricksNL2SQLAgent(cat, sch)

            agent = get_agent(catalog, schema)
            try:
                result = agent.query(prompt)
                response = result["answer"]

                # Show success/failure indicator
                if result["success"]:
                    st.markdown(response)
                else:
                    st.error(response)

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.caption("Built with Databricks Mosaic AI | LangChain | Streamlit")
