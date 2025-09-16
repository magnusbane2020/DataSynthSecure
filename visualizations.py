#!/usr/bin/env python3
"""
Interactive data visualization module for CrowdStrike opportunity analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OpportunityVisualizer:
    """Create interactive visualizations for opportunity analysis"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#FF4B4B',
            'secondary': '#FF6B6B', 
            'success': '#00CC88',
            'warning': '#FFB800',
            'info': '#0068C9',
            'muted': '#F0F2F6'
        }
    
    def create_scoring_distribution(self, scores_df):
        """Create interactive scoring distribution histogram"""
        fig = px.histogram(
            scores_df, 
            x='Score',
            nbins=20,
            title="Opportunity Scoring Distribution",
            labels={'Score': 'Score (0-100)', 'count': 'Number of Opportunities'},
            color_discrete_sequence=[self.color_palette['primary']]
        )
        
        # Add mean and median lines
        mean_score = scores_df['Score'].mean()
        median_score = scores_df['Score'].median()
        
        fig.add_vline(x=mean_score, line_dash="dash", line_color="orange", 
                     annotation_text=f"Mean: {mean_score:.1f}")
        fig.add_vline(x=median_score, line_dash="dash", line_color="green",
                     annotation_text=f"Median: {median_score:.1f}")
        
        fig.update_layout(
            showlegend=False,
            height=400,
            title_x=0.5,
            xaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_meddpicc_completion_chart(self, synthetic_df):
        """Create MEDDPICC completion pattern visualization"""
        if synthetic_df is None or 'Champion' not in synthetic_df.columns:
            return None
            
        # Calculate MEDDPICC completion rates
        meddpicc_fields = ['Metrics', 'Economic_Buyer', 'Decision_Criteria', 
                          'Decision_Process', 'Paper_Process', 'Identify_Pain', 
                          'Champion', 'Competition']
        
        completion_rates = {}
        for field in meddpicc_fields:
            if field in synthetic_df.columns:
                completion_rates[field] = synthetic_df[field].notna().mean() * 100
        
        # Create bar chart
        fig = px.bar(
            x=list(completion_rates.keys()),
            y=list(completion_rates.values()),
            title="MEDDPICC Field Completion Rates",
            labels={'x': 'MEDDPICC Fields', 'y': 'Completion Rate (%)'},
            color=list(completion_rates.values()),
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            height=400,
            title_x=0.5,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_value_vs_score_scatter(self, scores_df, synthetic_df):
        """Create opportunity value vs score scatter plot"""
        if synthetic_df is None or scores_df is None:
            return None
        
        # Validate merge keys
        if 'Opportunity_ID' not in scores_df.columns:
            st.warning("⚠️ Opportunity_ID missing from scores data. Cannot create value vs score chart.")
            return None
        if 'Opportunity_ID' not in synthetic_df.columns:
            st.warning("⚠️ Opportunity_ID missing from synthetic data. Cannot create value vs score chart.")
            return None
            
        # Merge dataframes
        merged = scores_df.merge(synthetic_df, on='Opportunity_ID', how='left')
        
        if 'Amount' not in merged.columns:
            st.info("ℹ️ Amount column not available. Cannot create value vs score scatter plot.")
            return None
        
        # Create scatter plot with safe hover data
        hover_data = []
        if 'Opportunity_Name' in merged.columns:
            hover_data.append('Opportunity_Name')
        if 'Account' in merged.columns:
            hover_data.append('Account')
            
        fig = px.scatter(
            merged,
            x='Score',
            y='Amount',
            hover_data=hover_data,
            title="Opportunity Value vs AI Score",
            labels={'Score': 'AI Score (0-100)', 'Amount': 'Opportunity Value ($)'},
            color='Score',
            color_continuous_scale='Viridis',
            size='Amount',
            size_max=15
        )
        
        fig.update_layout(
            height=500,
            title_x=0.5
        )
        
        return fig
    
    def create_pipeline_prioritization_matrix(self, scores_df, synthetic_df):
        """Create pipeline prioritization matrix"""
        if synthetic_df is None or scores_df is None:
            return None
        
        # Validate merge keys
        if 'Opportunity_ID' not in scores_df.columns or 'Opportunity_ID' not in synthetic_df.columns:
            st.warning("⚠️ Opportunity_ID missing from data. Cannot create prioritization matrix.")
            return None
            
        merged = scores_df.merge(synthetic_df, on='Opportunity_ID', how='left')
        
        if 'Amount' not in merged.columns or 'Champion' not in merged.columns:
            return None
        
        # Calculate true MEDDPICC completeness across all fields
        meddpicc_fields = ['Metrics', 'Economic_Buyer', 'Decision_Criteria', 
                          'Decision_Process', 'Paper_Process', 'Identify_Pain', 
                          'Champion', 'Competition']
        
        available_fields = [f for f in meddpicc_fields if f in merged.columns]
        if available_fields:
            merged['MEDDPICC_Complete'] = (merged[available_fields].notna().sum(axis=1) >= len(available_fields) * 0.7)
        else:
            merged['MEDDPICC_Complete'] = False
        merged['Value_Category'] = pd.cut(merged['Amount'], 
                                        bins=3, 
                                        labels=['Low Value', 'Medium Value', 'High Value'])
        merged['Score_Category'] = pd.cut(merged['Score'], 
                                        bins=[0, 40, 70, 100], 
                                        labels=['Low Score', 'Medium Score', 'High Score'])
        
        # Create bubble chart for prioritization with safe hover data
        hover_data = []
        for col in ['Opportunity_Name', 'Account', 'Stage']:
            if col in merged.columns:
                hover_data.append(col)
                
        fig = px.scatter(
            merged,
            x='Score',
            y='Amount',
            color='MEDDPICC_Complete',
            size='Amount',
            hover_data=hover_data,
            title="Pipeline Prioritization Matrix",
            labels={'Score': 'AI Score (0-100)', 'Amount': 'Opportunity Value ($)'},
            color_discrete_map={True: self.color_palette['success'], 
                              False: self.color_palette['warning']}
        )
        
        # Add quadrant lines
        fig.add_hline(y=merged['Amount'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=70, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant annotations
        fig.add_annotation(x=85, y=merged['Amount'].max() * 0.9, text="High Priority", 
                          showarrow=False, font=dict(size=12, color="green"))
        fig.add_annotation(x=25, y=merged['Amount'].max() * 0.9, text="Nurture", 
                          showarrow=False, font=dict(size=12, color="orange"))
        fig.add_annotation(x=85, y=merged['Amount'].min() * 1.5, text="Quick Wins", 
                          showarrow=False, font=dict(size=12, color="blue"))
        fig.add_annotation(x=25, y=merged['Amount'].min() * 1.5, text="Deprioritize", 
                          showarrow=False, font=dict(size=12, color="red"))
        
        fig.update_layout(
            height=500,
            title_x=0.5
        )
        
        return fig
    
    def create_stage_performance_chart(self, scores_df, synthetic_df):
        """Create performance by sales stage chart"""
        if synthetic_df is None or scores_df is None:
            return None
        
        # Validate merge keys
        if 'Opportunity_ID' not in scores_df.columns or 'Opportunity_ID' not in synthetic_df.columns:
            st.warning("⚠️ Opportunity_ID missing from data. Cannot create stage performance chart.")
            return None
            
        merged = scores_df.merge(synthetic_df, on='Opportunity_ID', how='left')
        
        if 'Stage Name' not in merged.columns:
            return None
        
        # Calculate average scores by stage with safe Amount handling
        agg_dict = {'Score': ['mean', 'count']}
        if 'Amount' in merged.columns:
            agg_dict['Amount'] = ['sum']
            
        stage_scores = merged.groupby('Stage Name').agg(agg_dict).round(2)
        
        if 'Amount' in merged.columns:
            stage_scores.columns = ['Avg_Score', 'Count', 'Total_Value']
        else:
            stage_scores.columns = ['Avg_Score', 'Count']
        stage_scores = stage_scores.reset_index()
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for average scores
        fig.add_trace(
            go.Bar(x=stage_scores['Stage Name'], y=stage_scores['Avg_Score'], 
                   name="Average Score", marker_color=self.color_palette['primary']),
            secondary_y=False,
        )
        
        # Add line chart for opportunity count
        fig.add_trace(
            go.Scatter(x=stage_scores['Stage Name'], y=stage_scores['Count'], 
                      mode='lines+markers', name="Opportunity Count",
                      line=dict(color=self.color_palette['info'])),
            secondary_y=True,
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Average Score", secondary_y=False)
        fig.update_yaxes(title_text="Number of Opportunities", secondary_y=True)
        
        fig.update_layout(
            title="Performance by Sales Stage",
            title_x=0.5,
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def display_key_metrics(self, scores_df, synthetic_df):
        """Display key metrics in a dashboard format"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if scores_df is not None and not scores_df.empty:
                avg_score = scores_df['Score'].mean()
                st.metric("Average Score", f"{avg_score:.1f}", 
                         delta=f"{avg_score - 50:.1f} vs baseline")
        
        with col2:
            if scores_df is not None and not scores_df.empty:
                high_score_count = len(scores_df[scores_df['Score'] > 70])
                total_count = len(scores_df)
                st.metric("High-Quality Opportunities", 
                         f"{high_score_count}/{total_count}",
                         delta=f"{(high_score_count/total_count)*100:.1f}%")
        
        with col3:
            if synthetic_df is not None and not synthetic_df.empty:
                total_value = synthetic_df['Amount'].sum() if 'Amount' in synthetic_df.columns else 0
                st.metric("Total Pipeline Value", 
                         f"${total_value:,.0f}")
        
        with col4:
            if synthetic_df is not None:
                # Calculate true MEDDPICC completeness (consistent with matrix logic)
                meddpicc_fields = ['Metrics', 'Economic_Buyer', 'Decision_Criteria', 
                                  'Decision_Process', 'Paper_Process', 'Identify_Pain', 
                                  'Champion', 'Competition']
                
                available_fields = [f for f in meddpicc_fields if f in synthetic_df.columns]
                if available_fields:
                    completeness = (synthetic_df[available_fields].notna().sum(axis=1) >= len(available_fields) * 0.7)
                    meddpicc_complete = completeness.sum()
                    total_opps = len(synthetic_df)
                    st.metric("MEDDPICC Complete (70%+)", 
                             f"{meddpicc_complete}/{total_opps}",
                             delta=f"{(meddpicc_complete/total_opps)*100:.1f}%")
                else:
                    st.metric("MEDDPICC Complete", "N/A", delta="No MEDDPICC fields found")

def display_interactive_visualizations():
    """Main function to display all visualizations"""
    st.header("📊 Interactive Data Visualizations")
    st.markdown("Analyze opportunity patterns and prioritize your pipeline")
    
    # Check for data availability
    scores_df = st.session_state.get('all_scores', st.session_state.get('test_scores'))
    synthetic_df = st.session_state.get('synthetic_data')
    
    if scores_df is None:
        st.warning("⚠️ Please complete AI scoring in Step 2 to view visualizations.")
        return
    
    if synthetic_df is None:
        st.warning("⚠️ Please generate synthetic data in Step 1 to view complete visualizations.")
    
    # Initialize visualizer
    visualizer = OpportunityVisualizer()
    
    # Display key metrics
    st.subheader("📈 Key Metrics Dashboard")
    visualizer.display_key_metrics(scores_df, synthetic_df)
    
    st.markdown("---")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Score Distribution", "MEDDPICC Analysis", "Pipeline Matrix", "Stage Performance"])
    
    with tab1:
        st.subheader("Score Distribution Analysis")
        fig1 = visualizer.create_scoring_distribution(scores_df)
        st.plotly_chart(fig1, use_container_width=True)
        
        if synthetic_df is not None:
            fig2 = visualizer.create_value_vs_score_scatter(scores_df, synthetic_df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("MEDDPICC Completion Analysis")
        if synthetic_df is not None:
            fig3 = visualizer.create_meddpicc_completion_chart(synthetic_df)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("MEDDPICC fields not available in current dataset.")
        else:
            st.info("Please generate synthetic data to view MEDDPICC analysis.")
    
    with tab3:
        st.subheader("Pipeline Prioritization Matrix")
        if synthetic_df is not None:
            fig4 = visualizer.create_pipeline_prioritization_matrix(scores_df, synthetic_df)
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)
                
                # Add interpretation guide
                st.markdown("""
                **How to interpret the Priority Matrix:**
                - **High Priority (Top Right)**: High scores + High value = Focus here first
                - **Quick Wins (Bottom Right)**: High scores + Lower value = Easy closes
                - **Nurture (Top Left)**: Low scores + High value = Needs work but worth it  
                - **Deprioritize (Bottom Left)**: Low scores + Low value = Minimal effort
                
                🟢 **Green dots** = MEDDPICC Complete | 🟡 **Yellow dots** = MEDDPICC Incomplete
                """)
            else:
                st.info("Unable to create prioritization matrix with current data.")
        else:
            st.info("Please generate synthetic data to view pipeline matrix.")
    
    with tab4:
        st.subheader("Performance by Sales Stage")
        if synthetic_df is not None:
            fig5 = visualizer.create_stage_performance_chart(scores_df, synthetic_df)
            if fig5:
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Unable to create stage performance chart with current data.")
        else:
            st.info("Please generate synthetic data to view stage analysis.")
    
    # Data export options
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if scores_df is not None:
            csv_data = scores_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Scores CSV",
                data=csv_data,
                file_name=f"opportunity_scores_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if synthetic_df is not None:
            csv_data = synthetic_df.to_csv(index=False)
            st.download_button(
                label="📋 Download Opportunities CSV", 
                data=csv_data,
                file_name=f"synthetic_opportunities_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )