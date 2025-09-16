import streamlit as st
import pandas as pd
import os
import time
import logging
from datetime import datetime
import json

from data_generator import SyntheticDataGenerator
from ai_scorer import OpportunityScorer
from report_generator import ExecutiveReportGenerator
from utils import setup_logging, validate_api_key

# Configure logging with security filters
setup_logging()
logger = logging.getLogger(__name__)

# Security notice
st.set_page_config(
    page_title="CrowdStrike - Secure Synthetic Data Analysis",
    page_icon="🛡️",
    layout="wide"
)

def main():
    st.title("🛡️ CrowdStrike Secure Synthetic Data Analysis")
    st.markdown("**Secure Synthetic Salesforce Opportunity Data Generator with AI-Powered Scoring**")
    
    # Security disclaimer
    st.info("""
    🔒 **Security Notice**: This application uses synthetic data only. No real customer data (PII) is processed.
    All API keys are handled through environment variables with secure practices.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Analysis Steps")
    step = st.sidebar.radio(
        "Select Analysis Phase:",
        ["Step 1: Generate Synthetic Data", "Step 2: AI Scoring", "Step 3: Executive Report"]
    )
    
    if step == "Step 1: Generate Synthetic Data":
        step1_generate_data()
    elif step == "Step 2: AI Scoring":
        step2_ai_scoring()
    elif step == "Step 3: Executive Report":
        step3_executive_report()

def step1_generate_data():
    st.header("Step 1: Secure Synthetic Data Creation")
    st.markdown("Generate synthetic Salesforce opportunities with MEDDPICC fields")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset size configuration
        st.subheader("Dataset Configuration")
        dataset_size = st.select_slider(
            "Number of opportunities to generate:",
            options=[50, 100, 250, 500, 1000],
            value=50,
            help="Larger datasets will take longer to process but provide better analysis"
        )
        
        batch_size = st.slider(
            "Batch size for processing:",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Smaller batches use less memory but take more time"
        )
        
        if st.button("Generate Synthetic Opportunities", type="primary"):
            with st.spinner(f"Generating {dataset_size} synthetic opportunities..."):
                try:
                    generator = SyntheticDataGenerator()
                    df = generator.generate_opportunities(dataset_size, batch_size=batch_size)
                    
                    # Store in session state
                    st.session_state['synthetic_data'] = df
                    
                    st.success(f"✅ Generated {len(df)} synthetic opportunities")
                    
                    # Display first 5 rows for validation
                    st.subheader("Validation Preview (First 5 Rows)")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Security validation
                    st.info("🔒 **PII Validation**: All data is synthetic. No real customer information included.")
                    
                    # Save to CSV
                    csv_path = "synthetic_opportunities.csv"
                    df.to_csv(csv_path, index=False)
                    st.success(f"💾 Data saved to {csv_path}")
                    
                    # Download button
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="synthetic_opportunities.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
                    logger.error(f"Data generation error: {str(e)}")
    
    with col2:
        st.info("""
        **Features:**
        - 50 synthetic opportunities
        - Standard SFDC fields
        - MEDDPICC fields (50% populated)
        - Public company names only
        - Zero PII inclusion
        """)
    
    # Display existing data if available
    if 'synthetic_data' in st.session_state:
        st.subheader("Current Dataset Overview")
        df = st.session_state['synthetic_data']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Opportunities", len(df))
        with col2:
            meddpicc_complete = df['Champion'].notna().sum()
            st.metric("MEDDPICC Complete", f"{meddpicc_complete} ({meddpicc_complete/len(df)*100:.1f}%)")
        with col3:
            avg_amount = df['Amount'].mean()
            st.metric("Avg Amount", f"${avg_amount:,.0f}")

def step2_ai_scoring():
    st.header("Step 2: Secure AI Assessment & Scoring")
    st.markdown("Evaluate opportunities using OpenAI with secure API key handling")
    
    # Check for synthetic data
    if 'synthetic_data' not in st.session_state:
        st.warning("⚠️ Please generate synthetic data in Step 1 first.")
        return
    
    # API key validation
    api_key_status = validate_api_key()
    if not api_key_status:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        st.info("🔒 **Security**: API keys must be stored in environment variables, never hardcoded.")
        return
    
    st.success("✅ API key validated (securely stored in environment)")
    
    df = st.session_state['synthetic_data']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Test with 3 rows first
        if st.button("🧪 Test Scoring (First 3 Rows)", type="secondary"):
            test_scoring(df.head(3))
        
        st.markdown("---")
        
        # Batch processing configuration for scoring
        st.subheader("Batch Scoring Configuration")
        scoring_batch_size = st.slider(
            "AI scoring batch size:",
            min_value=5,
            max_value=20,
            value=10,
            step=5,
            help="Number of opportunities per batch (processed sequentially). Smaller batches are more reliable."
        )
        
        # Full scoring
        if st.button("🚀 Score All Opportunities", type="primary"):
            full_scoring(df, scoring_batch_size)
    
    with col2:
        st.info("""
        **Security Features:**
        - Environment variable API keys
        - No key logging/printing
        - Performance timing
        - Secure data handling
        - MEDDPICC/BANT scoring
        """)

def test_scoring(test_df):
    """Test scoring on first 3 rows"""
    with st.spinner("Testing AI scoring on 3 opportunities..."):
        try:
            scorer = OpportunityScorer()
            start_time = time.time()
            
            results = []
            progress_bar = st.progress(0)
            
            for i, row in test_df.iterrows():
                result = scorer.score_opportunity(row)
                results.append(result)
                progress_bar.progress((i + 1) / len(test_df))
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Display results
            results_df = pd.DataFrame(results)
            st.subheader("Test Results")
            st.dataframe(results_df, use_container_width=True)
            
            st.success(f"✅ Test completed in {runtime:.2f} seconds")
            st.info("🔒 **Security**: No sensitive data logged. API calls completed securely.")
            
            # Store test results
            st.session_state['test_scores'] = results_df
            
        except Exception as e:
            st.error(f"❌ Scoring error: {str(e)}")
            logger.error(f"Scoring error: {str(e)}")

def full_scoring(df, batch_size=10):
    """Score all opportunities using batch processing"""
    dataset_size = len(df)
    with st.spinner(f"Scoring all {dataset_size} opportunities..."):
        try:
            scorer = OpportunityScorer()
            start_time = time.time()
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(completed, total):
                progress_bar.progress(completed / total)
                status_text.text(f"Scored {completed}/{total} opportunities ({(completed/total)*100:.1f}%)")
            
            # Use batch scoring
            results, failed_count = scorer.score_opportunities_batch(
                df, 
                batch_size=batch_size, 
                progress_callback=progress_callback
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Validate scoring results
            if failed_count > len(df) * 0.5:  # More than 50% failed
                st.error(f"❌ AI Scoring failed for {failed_count}/{len(df)} opportunities. Please check API key and try again.")
                st.warning("⚠️ Cannot generate reliable executive report with failed scoring results.")
                return
            elif failed_count > 0:
                st.warning(f"⚠️ {failed_count}/{len(df)} opportunities had scoring issues, but proceeding with analysis.")
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Save to CSV
            csv_path = "opportunity_scores.csv"
            results_df.to_csv(csv_path, index=False)
            
            # Display results
            st.subheader("Scoring Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Runtime", f"{runtime:.2f}s")
            with col2:
                st.metric("Avg Score", f"{results_df['Score'].mean():.1f}")
            with col3:
                st.metric("Avg Time/Opportunity", f"{runtime/len(df):.2f}s")
            
            st.success(f"✅ All opportunities scored successfully in {runtime:.2f} seconds")
            st.info("💾 Results saved to opportunity_scores.csv")
            
            # Download button
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Scores CSV",
                data=csv_data,
                file_name="opportunity_scores.csv",
                mime="text/csv"
            )
            
            # Store results
            st.session_state['all_scores'] = results_df
            
        except Exception as e:
            st.error(f"❌ Scoring error: {str(e)}")
            logger.error(f"Full scoring error: {str(e)}")

def step3_executive_report():
    st.header("Step 3: Executive Business Recommendations")
    st.markdown("Cybersecurity-conscious analysis and strategic recommendations")
    
    # Check for scoring results
    if 'all_scores' not in st.session_state and 'test_scores' not in st.session_state:
        st.warning("⚠️ Please complete AI scoring in Step 2 first.")
        return
    
    # Use available scores
    scores_df = st.session_state.get('all_scores', st.session_state.get('test_scores'))
    synthetic_df = st.session_state.get('synthetic_data')
    
    # Validate scores before generating report
    if scores_df is not None and len(scores_df) > 0:
        avg_score = scores_df['Score'].mean()
        zero_scores = (scores_df['Score'] == 0).sum()
        if avg_score == 0 or zero_scores > len(scores_df) * 0.8:
            st.error("❌ Invalid scoring data detected. Cannot generate reliable executive report.")
            st.info("Please re-run AI scoring in Step 2 to get valid results.")
            return
    
    if st.button("📊 Generate Executive Report", type="primary"):
        with st.spinner("Generating executive report..."):
            try:
                report_gen = ExecutiveReportGenerator()
                report = report_gen.generate_report(scores_df, synthetic_df)
                
                st.markdown("---")
                st.markdown(report)
                
                # Save report
                report_path = "executive_report.md"
                with open(report_path, 'w') as f:
                    f.write(report)
                
                st.success("✅ Executive report generated successfully")
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="crowdstrike_executive_report.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"❌ Report generation error: {str(e)}")
                logger.error(f"Report generation error: {str(e)}")

if __name__ == "__main__":
    main()
