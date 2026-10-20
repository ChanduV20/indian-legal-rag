
import streamlit as st
from hybrid_search import search_legal
from llm_explainer import explain_query

st.set_page_config(
    page_title="AI Legal Research",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #13263b, #254d68);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.35rem;
        font-size: 2rem;
    }

    .hero p {
        color: #dce6ef;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Header --------------------

st.markdown(
    """
    <div class="hero">
        <h1>⚖️ Legal AI</h1>
        <p>Indian Legal Research Assistant</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Sidebar --------------------

act_options = {
    "All Acts": None,
    "Bharatiya Nyaya Sanhita, 2023 (BNS)": "BNS",
    "Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)": "BNSS",
    "Bharatiya Sakshya Adhiniyam, 2023 (BSA)": "BSA",
}

with st.sidebar:
    st.header("Search settings")

    selected_label = st.selectbox(
        "Filter by Legal Act",
        list(act_options.keys()),
    )

    selected_act = act_options[selected_label]

    top_k = st.slider(
        "Number of results",
        min_value=3,
        max_value=10,
        value=5,
    )

    st.divider()
    st.subheader("About")
    st.caption(
        "Searches statutory text extracted from the BNS, "
        "BNSS and BSA source PDFs."
    )

# -------------------- Search --------------------

st.subheader("Search legal provisions")

st.write(
    "Describe the legal issue in ordinary language, "
    "or search for a section by number or title."
)

with st.form("legal_search_form"):
    question = st.text_input(
        "Your legal question",
        placeholder="e.g. What is the punishment for theft?",
    )

    submitted = st.form_submit_button(
        "Search provisions",
        type="primary",
        use_container_width=True,
    )

# -------------------- Results --------------------

if submitted:
    if not question.strip():
        st.warning("Enter a legal question to search.")
        st.stop()

    try:
        with st.spinner("Searching legal provisions..."):
            results = search_legal(
                question=question.strip(),
                top_k=top_k,
                act_filter=selected_act,
            )

        if not results:
            st.warning(
                "No matching provisions were found. "
                "Try rephrasing your question or selecting All Acts."
            )
            st.stop()

 

        st.subheader("Legal explanation")

        with st.spinner("Generating explanation..."):
            explanation = explain_query(question.strip(), results)

        if explanation and explanation.strip():
            st.markdown(explanation)
        else:
            st.warning(
                "The AI returned an empty explanation. "
                "Check your Gemini configuration and application logs."
            )

        st.divider()

        st.subheader("Retrieved legal provisions")

        for result in results:
            act_code = result.get("act_code", "Unknown Act")
            section_number = result.get("section_number", "?")
            section_title = result.get("section_title", "Untitled section")

            with st.expander(
                f"{act_code} §{section_number} — {section_title}",
                expanded=False,
            ):
                st.write(result.get("text", "No section text available."))

                start_page = result.get("start_page")
                end_page = result.get("end_page")

                if start_page is not None:
                    page_text = f"Source PDF page: {start_page}"
                    if end_page is not None and end_page != start_page:
                        page_text += f"–{end_page}"
                    st.caption(page_text)

    except Exception as exc:
        st.error("The search or explanation failed.")
        st.exception(exc)

# -------------------- Footer --------------------

st.divider()

st.caption(
    "Research assistance only. Verify applicable law, amendments, "
    "and authoritative sources before relying on a provision. "
    "This tool does not provide legal advice."
)
