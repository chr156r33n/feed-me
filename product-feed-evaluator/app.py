import os
import io
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from feed_parser import FeedParser
from evaluator_v2 import ProductFeedEvaluator


def _on_progress(update: dict):
    processed = update.get("processed", 0)
    total = update.get("total", 0)
    url = update.get("current_product_url", "")
    msg = update.get("message", "")
    st.session_state["progress_state"] = (processed, total, url, msg)


def main():
    st.set_page_config(page_title="Product Feed Evaluation Agent", layout="wide")

    load_dotenv()

    st.title("🧠 Product Feed Evaluation Agent")
    st.caption("Evaluate how well your Google Shopping feed answers buyer questions")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4-turbo"], index=0)
        locale = st.selectbox("Locale", ["en-GB", "en-US", "de-DE", "fr-FR", "es-ES"], index=0)
        num_questions = st.slider("Questions per product", min_value=4, max_value=24, value=12, step=1)
        batch_size = st.slider("Batch size", min_value=5, max_value=100, value=20, step=5)

        debug = st.checkbox("Debug mode (include error rows)", value=False)

        if st.button("Test OpenAI API"):
            if not api_key:
                st.error("No API key provided.")
            else:
                try:
                    # Lightweight ping using OpenAI client creation
                    from openai import OpenAI
                    _ = OpenAI(api_key=api_key)
                    st.success("OpenAI client initialized successfully.")
                except Exception as e:
                    st.error(f"OpenAI initialization failed: {e}")

        st.markdown("---")
        st.subheader("Selected fields")
        default_fields = ["title", "brand", "product_type", "google_product_category", "price", "availability", "color", "size", "material", "link", "description", "image_link"]
        all_fields = sorted(list(set(default_fields + FeedParser().recommended_fields + FeedParser().required_fields)))
        selected_fields = st.multiselect("Fields used to build context", options=all_fields, default=default_fields)

    st.subheader("Input Feed")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload Google Shopping XML feed", type=["xml"]) 
    with col2:
        sample_btn = st.button("Use included sample feed.xml")

    xml_content = None
    if uploaded is not None:
        xml_content = uploaded.read().decode("utf-8", errors="replace")
    elif sample_btn:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_feed.xml")
        with open(sample_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

    if not xml_content:
        st.info("Upload a feed or click 'Use included sample feed.xml' to get started.")
        return

    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar.")
        return

    parser = FeedParser()
    with st.spinner("Parsing feed…"):
        try:
            products = parser.parse_xml_feed(xml_content)
            products_df = pd.DataFrame(products)
        except Exception as e:
            st.error(f"Failed to parse XML feed: {e}")
            return

    if products_df.empty:
        st.warning("No products found in the feed.")
        return

    st.success(f"Parsed {len(products_df)} products")
    st.dataframe(products_df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Evaluate")
    start = st.button("Start Evaluation")

    if start:
        evaluator = ProductFeedEvaluator(api_key=api_key, model=model, locale=locale)
        progress_bar = st.progress(0)
        status = st.empty()

        def on_progress_local(update: dict):
            _on_progress(update)
            processed, total, url, msg = st.session_state.get("progress_state", (0, len(products_df), "", ""))
            if total:
                progress_bar.progress(min(processed / total, 1.0))
            status.write(f"Processed {processed}/{total} {('- ' + url) if url else ''}")

        with st.spinner("Running evaluation (this may take a while)…"):
            results_df = evaluator.evaluate_products_batch(
                products_df=products_df,
                selected_fields=selected_fields,
                num_questions=num_questions,
                batch_size=batch_size,
                on_progress=on_progress_local,
                debug=debug,
            )

        if results_df.empty:
            st.error("Evaluation returned no rows. Enable Debug mode to capture errors, and verify your OpenAI API key.")
        else:
            st.success("Evaluation complete")
            st.dataframe(results_df.head(50), use_container_width=True)

        # CSV download
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name="feed_analysis.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

