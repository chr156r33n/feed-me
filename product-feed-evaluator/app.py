"""
Main Streamlit application for the Product Feed Evaluation Agent.
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import requests
from feed_parser import FeedParser
from evaluator import ProductFeedEvaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Product Feed Evaluator",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def main():
    st.title("🧠 Product Feed Evaluation Agent")
    st.markdown("Evaluate Google Shopping product feed quality by checking how well each product answers buyer questions.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key. You can also set it in your environment variables."
        )
        
        if not api_key:
            st.error("Please enter your OpenAI API key to continue.")
            return
        
        # Model selection
        model = st.selectbox(
            "GPT Model",
            ["gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
            index=0,
            help="gpt-4o-mini is faster and cheaper, gpt-4-turbo is more accurate"
        )
        
        # Number of questions
        num_questions = st.slider(
            "Questions per Product",
            min_value=5,
            max_value=20,
            value=12,
            help="Number of buyer questions to generate per product"
        )
        
        # Batch size
        batch_size = st.slider(
            "Batch Size",
            min_value=5,
            max_value=50,
            value=20,
            help="Number of products to process in each batch"
        )
        
        # Debug mode toggle
        debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Show verbose logs and detailed progress while running"
        )
        
        # Field selection
        st.subheader("📋 Field Selection")
        st.markdown("Select which fields to include in product context:")
        
        default_fields = ['title', 'description', 'brand', 'price', 'availability', 'image_link']
        available_fields = [
            # Identifiers and links
            'id', 'title', 'description', 'link', 'mobile_link',
            # Media
            'image_link', 'additional_image_link',
            # Availability and condition
            'availability', 'availability_date', 'expiration_date', 'condition',
            # Pricing
            'price', 'sale_price', 'sale_price_effective_date', 'cost_of_goods_sold',
            # Brand and product identifiers
            'brand', 'gtin', 'mpn', 'identifier_exists', 'item_group_id',
            # Taxonomy and type
            'google_product_category', 'product_type',
            # Variants and attributes
            'color', 'size', 'size_type', 'size_system', 'material', 'pattern', 'age_group', 'gender', 'adult', 'multipack', 'is_bundle',
            # Energy labels
            'energy_efficiency_class', 'min_energy_efficiency_class', 'max_energy_efficiency_class',
            # Unit pricing
            'unit_pricing_measure', 'unit_pricing_base_measure',
            # Shipping and tax
            'shipping', 'shipping_weight', 'shipping_length', 'shipping_width', 'shipping_height', 'shipping_label', 'tax',
            # Campaign labels and destinations
            'custom_label_0', 'custom_label_1', 'custom_label_2', 'custom_label_3', 'custom_label_4', 'included_destination', 'excluded_destination', 'shopping_ads_excluded_country',
            # Programs
            'loyalty_points', 'installment', 'subscription_cost'
        ]
        
        selected_fields = st.multiselect(
            "Fields to include",
            available_fields,
            default=default_fields,
            help="The 'link' field is always included automatically"
        )
        
        if 'link' not in selected_fields:
            selected_fields.append('link')
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📥 Input Feed")
        
        # Feed input method
        input_method = st.radio(
            "Choose input method:",
            ["Upload XML file", "Paste XML content", "Enter feed URL"]
        )
        
        xml_content = None
        
        if input_method == "Upload XML file":
            uploaded_file = st.file_uploader(
                "Upload your Google Shopping XML feed",
                type=['xml'],
                help="Upload a valid Google Shopping XML feed file"
            )
            if uploaded_file:
                xml_content = uploaded_file.read().decode('utf-8')
        
        elif input_method == "Paste XML content":
            xml_content = st.text_area(
                "Paste your XML feed content",
                height=200,
                help="Paste the raw XML content of your feed"
            )
        
        elif input_method == "Enter feed URL":
            feed_url = st.text_input(
                "Enter feed URL",
                placeholder="https://example.com/feed.xml",
                help="Enter the public URL of your XML feed"
            )
            if feed_url:
                try:
                    response = requests.get(feed_url, timeout=30)
                    response.raise_for_status()
                    xml_content = response.text
                    st.success("Feed loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading feed: {e}")
    
    with col2:
        st.header("📊 Quick Stats")
        
        if xml_content:
            try:
                parser = FeedParser()
                products = parser.parse_xml_feed(xml_content)
                
                st.metric("Total Products", len(products))
                
                # Show field coverage
                field_coverage = {}
                # Reflect selected fields in Quick Stats (exclude 'link')
                stats_fields = [f for f in selected_fields if f != 'link'] if 'selected_fields' in locals() else []
                if not stats_fields:
                    stats_fields = ['title', 'description', 'brand', 'price', 'availability']
                max_stats = 10
                display_fields = stats_fields[:max_stats]
                for field in display_fields:
                    count = sum(1 for p in products if p.get(field))
                    coverage = round(count / len(products) * 100, 1) if products else 0
                    field_coverage[field] = coverage
                
                st.subheader("Field Coverage")
                for field, coverage in field_coverage.items():
                    st.metric(field.replace('_', ' ').title(), f"{coverage}%")
                if len(stats_fields) > max_stats:
                    st.caption(f"Showing first {max_stats} of {len(stats_fields)} selected fields")
                
            except Exception as e:
                st.error(f"Error parsing feed: {e}")
    
    # Process button
    if xml_content and selected_fields:
        if st.button("🚀 Start Evaluation", disabled=st.session_state.processing):
            st.session_state.processing = True
            
            try:
                # Initialize components
                parser = FeedParser()
                evaluator = ProductFeedEvaluator(api_key, model)
                
                # Process feed
                with st.spinner("Processing feed..."):
                    products_df = parser.process_feed(xml_content, selected_fields)
                
                st.success(f"Feed processed: {len(products_df)} products found")
                
                # Evaluate products
                with st.spinner("Evaluating products with AI..."):
                    total_products = len(products_df)
                    progress_bar = st.progress(0, text="Starting evaluation...")
                    status_text = st.empty()
                    debug_log_placeholder = None
                    if debug_mode:
                        st.session_state["debug_lines"] = []
                        debug_expander = st.expander("🪲 Debug Logs", expanded=False)
                        debug_log_placeholder = debug_expander.empty()

                    def on_progress_update(state):
                        processed = state.get("processed", 0)
                        total = state.get("total", total_products)
                        current_url = state.get("current_product_url", "")
                        pct = int(processed / total * 100) if total > 0 else 0
                        progress_bar.progress(pct, text=f"Evaluating products... {processed}/{total}")
                        if current_url:
                            status_text.write(f"Current: {current_url}")
                        if debug_mode and debug_log_placeholder is not None:
                            message = state.get("message", "")
                            if message:
                                st.session_state["debug_lines"].append(message)
                                debug_log_placeholder.code("\n".join(st.session_state["debug_lines"]))

                    results_df = evaluator.evaluate_products_batch(
                        products_df,
                        selected_fields,
                        num_questions,
                        batch_size,
                        on_progress=on_progress_update,
                        debug=debug_mode
                    )
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d")
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/feed_analysis_{timestamp}.csv"
                results_df.to_csv(output_file, index=False)
                
                st.session_state.results = results_df
                st.success(f"Evaluation complete! Results saved to {output_file}")
                
            except Exception as e:
                st.error(f"Error during evaluation: {e}")
            finally:
                st.session_state.processing = False
    
    # Display results
    if st.session_state.results is not None:
        st.header("📈 Evaluation Results")
        
        results_df = st.session_state.results
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_coverage = results_df['coverage_pct'].mean()
            st.metric("Average Coverage", f"{avg_coverage:.1f}%")
        
        with col2:
            high_coverage = len(results_df[results_df['coverage_pct'] >= 70])
            st.metric("High Coverage (≥70%)", high_coverage)
        
        with col3:
            low_coverage = len(results_df[results_df['coverage_pct'] < 30])
            st.metric("Low Coverage (<30%)", low_coverage)
        
        with col4:
            total_products = len(results_df)
            st.metric("Total Products", total_products)
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            min_coverage = st.slider("Minimum Coverage %", 0, 100, 0)
        with col2:
            sort_by = st.selectbox("Sort by", ["coverage_pct", "product_url", "yes_count"])
        
        # Filter and sort results
        filtered_results = results_df[results_df['coverage_pct'] >= min_coverage]
        filtered_results = filtered_results.sort_values(sort_by, ascending=False)
        
        # Display table
        st.dataframe(
            filtered_results,
            use_container_width=True,
            column_config={
                "product_url": st.column_config.LinkColumn("Product URL"),
                "coverage_pct": st.column_config.NumberColumn("Coverage %", format="%.1f%%"),
                "yes_count": "Yes",
                "partial_count": "Partial", 
                "no_count": "No"
            }
        )
        
        # Download button
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"feed_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Analysis insights
        st.subheader("🔍 Analysis Insights")
        
        # Coverage distribution
        coverage_ranges = [
            (0, 30, "Poor"),
            (30, 50, "Fair"), 
            (50, 70, "Good"),
            (70, 100, "Excellent")
        ]
        
        for min_val, max_val, label in coverage_ranges:
            count = len(results_df[
                (results_df['coverage_pct'] >= min_val) & 
                (results_df['coverage_pct'] < max_val)
            ])
            percentage = count / len(results_df) * 100 if len(results_df) > 0 else 0
            st.metric(f"{label} Coverage", f"{count} products ({percentage:.1f}%)")
        
        # Most common missing fields
        if 'missing_core_fields' in results_df.columns:
            missing_fields = results_df['missing_core_fields'].str.split(', ').explode()
            missing_counts = missing_fields.value_counts()
            if not missing_counts.empty:
                st.subheader("🚨 Most Common Missing Fields")
                for field, count in missing_counts.head().items():
                    if field:  # Skip empty strings
                        st.write(f"• {field}: {count} products")

if __name__ == "__main__":
    main()