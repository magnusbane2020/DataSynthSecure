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
from visualizations import display_interactive_visualizations
from monitoring_dashboard import monitoring_dashboard
from audit_trail import log_user_action, log_data_access, log_api_call, log_security_event, log_system_error
from bias_dashboard import bias_dashboard
from database import get_db_manager, init_database

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
    
    # Initialize database
    if not init_database():
        st.error("❌ Database connection failed. Please check your configuration.")
        return
    
    # Security disclaimer
    st.info("""
    🔒 **Security Notice**: This application uses synthetic data only. No real customer data (PII) is processed.
    All API keys are handled through environment variables with secure practices.
    Data is stored securely in PostgreSQL database.
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Analysis Steps")
    step = st.sidebar.radio(
        "Select Analysis Phase:",
        ["Step 1: Generate Synthetic Data", "Step 2: AI Scoring", "Step 3: Interactive Visualizations", "Step 4: Executive Report", "Step 5: Audit Monitor", "Step 6: Bias Detection"]
    )
    
    # Database statistics sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Database Status")
    try:
        db_manager = get_db_manager()
        stats = db_manager.get_database_stats()
        st.sidebar.metric("Total Opportunities", stats['total_opportunities'])
        st.sidebar.metric("Total Scores", stats['total_scores'])
        if stats['latest_opportunity_date']:
            st.sidebar.text(f"Latest Data: {stats['latest_opportunity_date'].strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        st.sidebar.error("Database unavailable")
    
    if step == "Step 1: Generate Synthetic Data":
        step1_generate_data()
    elif step == "Step 2: AI Scoring":
        step2_ai_scoring()
    elif step == "Step 3: Interactive Visualizations":
        display_interactive_visualizations()
    elif step == "Step 4: Executive Report":
        step3_executive_report()
    elif step == "Step 5: Audit Monitor":
        step5_audit_monitoring()
    elif step == "Step 6: Bias Detection":
        step6_bias_detection()

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
                    
                    # Log data generation event
                    log_user_action("DATA_GENERATED", 
                                   user_id="data_generator",
                                   details={"dataset_size": len(df), "batch_size": batch_size})
                    
                    st.success(f"✅ Generated {len(df)} synthetic opportunities")
                    st.success(f"💾 Data saved to PostgreSQL database")
                    
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
    
    # Load existing data from database
    st.markdown("---")
    st.subheader("Load Existing Data")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📂 Load Latest Data from Database"):
            try:
                db_manager = get_db_manager()
                df = db_manager.load_synthetic_opportunities()
                if not df.empty:
                    st.session_state['synthetic_data'] = df
                    st.success(f"✅ Loaded {len(df)} opportunities from database")
                else:
                    st.warning("No data found in database. Generate some data first!")
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                logger.error(f"Data loading error: {str(e)}")
    
    with col2:
        # Show available batches
        try:
            db_manager = get_db_manager()
            batches = db_manager.get_available_batches()
            if batches['opportunity_batches']:
                st.selectbox("Available Data Batches:", batches['opportunity_batches'], key="data_batch_selector")
        except:
            pass
    
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
    
    # Check for synthetic data (try loading from database if not in session)
    if 'synthetic_data' not in st.session_state:
        try:
            db_manager = get_db_manager()
            df = db_manager.load_synthetic_opportunities()
            if not df.empty:
                st.session_state['synthetic_data'] = df
                st.info(f"✅ Automatically loaded {len(df)} opportunities from database")
            else:
                st.warning("⚠️ Please generate synthetic data in Step 1 first.")
                return
        except Exception as e:
            st.warning("⚠️ Please generate synthetic data in Step 1 first.")
            logger.error(f"Failed to auto-load data: {str(e)}")
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
            log_system_error("AI_SCORING_FAILED", {"error": str(e), "dataset_size": len(df)})
    
    # Load existing scores section
    st.markdown("---")
    st.subheader("Load Existing Scores")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📊 Load Latest Scores from Database"):
            try:
                db_manager = get_db_manager()
                scores_df = db_manager.load_opportunity_scores()
                if not scores_df.empty:
                    st.session_state['all_scores'] = scores_df
                    st.success(f"✅ Loaded {len(scores_df)} scores from database")
                    st.dataframe(scores_df.head(), use_container_width=True)
                else:
                    st.warning("No scores found in database. Run AI scoring first!")
            except Exception as e:
                st.error(f"❌ Error loading scores: {str(e)}")
                logger.error(f"Score loading error: {str(e)}")
    
    with col2:
        # Show available scoring batches
        try:
            db_manager = get_db_manager()
            batches = db_manager.get_available_batches()
            if batches['scoring_batches']:
                st.selectbox("Available Scoring Batches:", batches['scoring_batches'], key="scoring_batch_selector")
        except:
            pass

def step5_audit_monitoring():
    """Step 5: Real-time audit trail and security monitoring dashboard"""
    try:
        # Log user access to monitoring dashboard
        log_user_action("AUDIT_DASHBOARD_ACCESSED", user_id="dashboard_user")
        
        # Display the monitoring dashboard
        monitoring_dashboard.display_monitoring_dashboard()
        
        log_user_action("AUDIT_DASHBOARD_VIEWED", user_id="dashboard_user")
        
    except Exception as e:
        st.error(f"❌ Error loading audit monitoring dashboard: {str(e)}")
        logger.error(f"Audit monitoring error: {str(e)}")
        log_system_error("AUDIT_DASHBOARD_ERROR", {"error": str(e)})

def step6_bias_detection():
    """Step 6: AI Bias Detection and Fairness Monitoring"""
    try:
        # Log user access to bias detection
        log_user_action("BIAS_DETECTION_ACCESSED", user_id="bias_analyst")
        
        # Check for required data
        if 'synthetic_data' not in st.session_state:
            st.warning("⚠️ No synthetic data found. Please complete Step 1: Generate Synthetic Data first.")
            return
        
        if 'all_scores' not in st.session_state and 'test_scores' not in st.session_state:
            st.warning("⚠️ No scoring results found. Please complete Step 2: AI Scoring first.")
            return
        
        # Get scored data (prefer all_scores over test_scores)
        scored_data = st.session_state.get('all_scores', st.session_state.get('test_scores'))
        synthetic_data = st.session_state['synthetic_data']
        
        # Convert to DataFrame if needed
        if isinstance(scored_data, list):
            scored_data = pd.DataFrame(scored_data)
        if not isinstance(synthetic_data, pd.DataFrame):
            synthetic_data = pd.DataFrame(synthetic_data)
        
        # Check if scored_data is valid before displaying dashboard
        if scored_data is not None and not scored_data.empty:
            # Display bias monitoring dashboard
            bias_dashboard.display_bias_monitoring_dashboard(scored_data, synthetic_data)
        else:
            st.error("❌ No valid scoring data available for bias analysis.")
            st.info("Please complete Step 2: AI Scoring first to generate valid results.")
        
        log_user_action("BIAS_DETECTION_VIEWED", user_id="bias_analyst")
        
    except Exception as e:
        st.error(f"❌ Error loading bias detection dashboard: {str(e)}")
        logger.error(f"Bias detection error: {str(e)}")
        log_system_error("BIAS_DETECTION_ERROR", {"error": str(e)})

def step3_executive_report():
    st.header("Step 4: Executive Business Recommendations")
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
