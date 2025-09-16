"""
Bias Monitoring Dashboard for AI Scoring Fairness
Provides interactive visualization and monitoring of bias detection results,
explainability metrics, and mitigation recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta

from bias_detection import bias_detector, BiasType, FairnessLevel, BiasReport
from audit_trail import log_user_action

class BiasMonitoringDashboard:
    """Interactive dashboard for bias detection and fairness monitoring"""
    
    def __init__(self):
        self.color_palette = {
            'fair': '#2ca02c',           # Green
            'attention': '#ff7f0e',      # Orange  
            'bias_detected': '#d62728',  # Red
            'critical': '#8b0000',       # Dark Red
            'primary': '#1f77b4',        # Blue
            'secondary': '#7f7f7f'       # Gray
        }
        
        self.fairness_colors = {
            FairnessLevel.FAIR: self.color_palette['fair'],
            FairnessLevel.ATTENTION_NEEDED: self.color_palette['attention'],
            FairnessLevel.BIAS_DETECTED: self.color_palette['bias_detected'],
            FairnessLevel.CRITICAL_BIAS: self.color_palette['critical']
        }
    
    def display_bias_monitoring_dashboard(self, scored_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Main bias monitoring dashboard"""
        st.header("🔍 AI Bias Detection & Fairness Monitoring")
        st.markdown("Comprehensive fairness validation and explainability analysis for AI scoring")
        
        # Check if we have sufficient data
        if scored_data.empty or synthetic_data.empty:
            st.warning("⚠️ No scored data available for bias analysis. Please complete AI scoring first.")
            return
        
        # Dashboard controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Full Bias Analysis", "Quick Fairness Check", "Segment Analysis", "Temporal Analysis"],
                help="Select the type of bias analysis to perform"
            )
        
        with col2:
            min_sample_size = st.slider(
                "Minimum Sample Size",
                min_value=10,
                max_value=100,
                value=20,
                help="Minimum samples required per segment for reliable analysis"
            )
        
        with col3:
            if st.button("🔍 Run Bias Analysis", type="primary"):
                with st.spinner("Analyzing scoring fairness..."):
                    self._run_bias_analysis(scored_data, synthetic_data, analysis_type, min_sample_size)
        
        # Display cached results if available
        if 'bias_report' in st.session_state:
            self._display_bias_analysis_results(st.session_state['bias_report'])
        else:
            st.info("👆 Click 'Run Bias Analysis' to start fairness evaluation")
    
    def _run_bias_analysis(self, scored_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                          analysis_type: str, min_sample_size: int):
        """Execute bias analysis and store results"""
        try:
            # Log bias analysis start
            log_user_action("BIAS_ANALYSIS_STARTED", 
                          details={"analysis_type": analysis_type, "dataset_size": len(scored_data)})
            
            # Filter data by minimum sample size if needed
            if analysis_type == "Segment Analysis":
                # Ensure segments have minimum sample size
                filtered_synthetic = self._filter_by_sample_size(synthetic_data, min_sample_size)
                if len(filtered_synthetic) < len(synthetic_data):
                    st.info(f"Filtered dataset to ensure minimum {min_sample_size} samples per segment")
                synthetic_data = filtered_synthetic
            
            # Run bias detection
            bias_report = bias_detector.detect_bias(scored_data, synthetic_data)
            
            # Store results in session state
            st.session_state['bias_report'] = bias_report
            st.session_state['bias_analysis_timestamp'] = datetime.now()
            
            # Show success message
            fairness_emoji = {
                FairnessLevel.FAIR: "✅",
                FairnessLevel.ATTENTION_NEEDED: "⚠️", 
                FairnessLevel.BIAS_DETECTED: "🚨",
                FairnessLevel.CRITICAL_BIAS: "🔴"
            }
            
            emoji = fairness_emoji.get(bias_report.overall_fairness, "📊")
            st.success(f"{emoji} Bias analysis completed! Overall fairness: {bias_report.overall_fairness.value}")
            
        except Exception as e:
            st.error(f"❌ Bias analysis failed: {str(e)}")
            st.exception(e)
    
    def _filter_by_sample_size(self, df: pd.DataFrame, min_size: int) -> pd.DataFrame:
        """Filter dataset to ensure minimum sample size per segment"""
        # This is a simplified approach - in practice, you'd implement more sophisticated filtering
        return df  # For now, return original dataset
    
    def _display_bias_analysis_results(self, bias_report: BiasReport):
        """Display comprehensive bias analysis results"""
        
        # Overall fairness status
        self._display_fairness_overview(bias_report)
        
        # Bias metrics visualization
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Fairness Overview",
            "⚖️ Bias Metrics", 
            "📈 Score Analysis",
            "🎯 Recommendations",
            "📋 Detailed Report"
        ])
        
        with tab1:
            self._display_fairness_overview_tab(bias_report)
        
        with tab2:
            self._display_bias_metrics_tab(bias_report)
        
        with tab3:
            self._display_score_analysis_tab(bias_report)
        
        with tab4:
            self._display_recommendations_tab(bias_report)
        
        with tab5:
            self._display_detailed_report_tab(bias_report)
    
    def _display_fairness_overview(self, bias_report: BiasReport):
        """Display high-level fairness status"""
        st.subheader("🎯 Fairness Assessment")
        
        # Fairness status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fairness_color = self.fairness_colors[bias_report.overall_fairness]
            st.markdown(f"""
            <div style="background-color: {fairness_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {fairness_color};">
                <h3 style="color: {fairness_color}; margin: 0;">Overall Fairness</h3>
                <p style="margin: 0; font-size: 1.2em; font-weight: bold;">{bias_report.overall_fairness.value.title()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            bias_count = sum(1 for metric in bias_report.bias_metrics if metric.is_biased)
            color = self.color_palette['bias_detected'] if bias_count > 0 else self.color_palette['fair']
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {color};">
                <h3 style="color: {color}; margin: 0;">Bias Metrics</h3>
                <p style="margin: 0; font-size: 1.2em; font-weight: bold;">{bias_count} / {len(bias_report.bias_metrics)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            confidence_color = self.color_palette['fair'] if bias_report.confidence_score > 0.7 else self.color_palette['attention']
            st.markdown(f"""
            <div style="background-color: {confidence_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {confidence_color};">
                <h3 style="color: {confidence_color}; margin: 0;">Confidence</h3>
                <p style="margin: 0; font-size: 1.2em; font-weight: bold;">{bias_report.confidence_score:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sig_color = self.color_palette['fair'] if bias_report.statistical_significance else self.color_palette['secondary']
            st.markdown(f"""
            <div style="background-color: {sig_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {sig_color};">
                <h3 style="color: {sig_color}; margin: 0;">Statistical Sig.</h3>
                <p style="margin: 0; font-size: 1.2em; font-weight: bold;">{'Yes' if bias_report.statistical_significance else 'No'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _display_fairness_overview_tab(self, bias_report: BiasReport):
        """Display fairness overview visualizations"""
        
        # Bias metrics by type
        bias_by_type = {}
        for metric in bias_report.bias_metrics:
            bias_type = metric.bias_type.value
            if bias_type not in bias_by_type:
                bias_by_type[bias_type] = {'total': 0, 'biased': 0}
            bias_by_type[bias_type]['total'] += 1
            if metric.is_biased:
                bias_by_type[bias_type]['biased'] += 1
        
        # Create bias type summary chart
        if bias_by_type:
            bias_types = list(bias_by_type.keys())
            bias_counts = [bias_by_type[bt]['biased'] for bt in bias_types]
            total_counts = [bias_by_type[bt]['total'] for bt in bias_types]
            
            fig_bias_types = go.Figure(data=[
                go.Bar(name='Biased', x=bias_types, y=bias_counts, marker_color=self.color_palette['bias_detected']),
                go.Bar(name='Fair', x=bias_types, y=[total_counts[i] - bias_counts[i] for i in range(len(bias_counts))], marker_color=self.color_palette['fair'])
            ])
            
            fig_bias_types.update_layout(
                title="Bias Detection by Type",
                barmode='stack',
                height=400,
                xaxis_title="Bias Type",
                yaxis_title="Number of Metrics"
            )
            
            st.plotly_chart(fig_bias_types, use_container_width=True)
        
        # Fairness level distribution
        fairness_levels = [metric.fairness_level.value for metric in bias_report.bias_metrics]
        fairness_counts = pd.Series(fairness_levels).value_counts()
        
        if not fairness_counts.empty:
            colors = [self.fairness_colors.get(FairnessLevel(level), self.color_palette['secondary']) 
                     for level in fairness_counts.index]
            
            fig_fairness = px.pie(
                values=fairness_counts.values,
                names=fairness_counts.index,
                title="Fairness Level Distribution",
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig_fairness, use_container_width=True)
    
    def _display_bias_metrics_tab(self, bias_report: BiasReport):
        """Display detailed bias metrics"""
        st.subheader("⚖️ Detailed Bias Metrics")
        
        if not bias_report.bias_metrics:
            st.info("No bias metrics to display")
            return
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            show_biased_only = st.checkbox("Show only biased metrics", value=False)
        with col2:
            bias_type_filter = st.multiselect(
                "Filter by bias type",
                options=[bt.value for bt in BiasType],
                default=[bt.value for bt in BiasType]
            )
        
        # Filter metrics
        filtered_metrics = bias_report.bias_metrics
        if show_biased_only:
            filtered_metrics = [m for m in filtered_metrics if m.is_biased]
        if bias_type_filter:
            filtered_metrics = [m for m in filtered_metrics if m.bias_type.value in bias_type_filter]
        
        # Display metrics table
        if filtered_metrics:
            metrics_data = []
            for metric in filtered_metrics:
                metrics_data.append({
                    'Metric': metric.metric_name,
                    'Type': metric.bias_type.value,
                    'Value': f"{metric.value:.3f}",
                    'Threshold': f"{metric.threshold:.3f}",
                    'Status': '🚨 Biased' if metric.is_biased else '✅ Fair',
                    'Fairness Level': metric.fairness_level.value,
                    'Segments': ', '.join(metric.segments[:3]) + ('...' if len(metric.segments) > 3 else '')
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Detailed metric views
            st.subheader("📊 Metric Details")
            selected_metric = st.selectbox(
                "Select metric for detailed view",
                options=[(i, m.metric_name) for i, m in enumerate(filtered_metrics)],
                format_func=lambda x: x[1]
            )
            
            if selected_metric:
                metric = filtered_metrics[selected_metric[0]]
                self._display_metric_details(metric)
        else:
            st.info("No metrics match the selected filters")
    
    def _display_metric_details(self, metric):
        """Display detailed view of a specific bias metric"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Metric:** {metric.metric_name}")
            st.markdown(f"**Type:** {metric.bias_type.value}")
            st.markdown(f"**Value:** {metric.value:.4f}")
            st.markdown(f"**Threshold:** {metric.threshold:.4f}")
            st.markdown(f"**Is Biased:** {'Yes' if metric.is_biased else 'No'}")
            st.markdown(f"**Fairness Level:** {metric.fairness_level.value}")
        
        with col2:
            st.markdown(f"**Segments Analyzed:** {', '.join(metric.segments)}")
            st.markdown(f"**Recommendation:**")
            st.info(metric.recommendation)
        
        # Display metric-specific details
        if metric.details:
            st.subheader("📋 Technical Details")
            
            # Handle different types of details
            if 'acceptance_rates' in metric.details:
                # Demographic parity details
                rates = metric.details['acceptance_rates']
                if rates:
                    fig = px.bar(
                        x=list(rates.keys()),
                        y=list(rates.values()),
                        title=f"Acceptance Rates by Segment",
                        labels={'x': 'Segment', 'y': 'Acceptance Rate'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif 'segment_means' in metric.details:
                # Score distribution details
                means = metric.details['segment_means']
                if means:
                    fig = px.bar(
                        x=list(means.keys()),
                        y=list(means.values()),
                        title=f"Mean Scores by Segment",
                        labels={'x': 'Segment', 'y': 'Mean Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Raw details
            with st.expander("Raw Technical Details"):
                st.json(metric.details)
    
    def _display_score_analysis_tab(self, bias_report: BiasReport):
        """Display score distribution analysis"""
        st.subheader("📈 Score Distribution Analysis")
        
        # This would require access to the underlying score data
        # For now, show placeholder for score analysis
        st.info("Score distribution analysis requires access to underlying scoring data. This feature will be enhanced with direct data integration.")
        
        # Show summary statistics from bias metrics
        distribution_metrics = [m for m in bias_report.bias_metrics if m.bias_type == BiasType.SCORE_DISTRIBUTION]
        
        if distribution_metrics:
            st.subheader("Distribution Differences by Segment")
            
            for metric in distribution_metrics:
                if 'segment_means' in metric.details:
                    segment_means = metric.details['segment_means']
                    segment_stds = metric.details.get('segment_stds', {})
                    
                    # Create distribution comparison chart
                    fig = go.Figure()
                    
                    segments = list(segment_means.keys())
                    means = list(segment_means.values())
                    stds = [segment_stds.get(seg, 0) for seg in segments]
                    
                    fig.add_trace(go.Bar(
                        x=segments,
                        y=means,
                        error_y=dict(type='data', array=stds),
                        name='Mean Score ± Std Dev',
                        marker_color=self.color_palette['primary']
                    ))
                    
                    fig.update_layout(
                        title=f"Score Distribution: {metric.metric_name}",
                        xaxis_title="Segment",
                        yaxis_title="Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def _display_recommendations_tab(self, bias_report: BiasReport):
        """Display bias mitigation recommendations"""
        st.subheader("🎯 Bias Mitigation Recommendations")
        
        if not bias_report.mitigation_recommendations:
            st.success("✅ No bias mitigation recommendations needed - scoring appears fair!")
            return
        
        # Priority recommendations
        st.markdown("### High Priority Actions")
        critical_metrics = [m for m in bias_report.bias_metrics if m.fairness_level == FairnessLevel.CRITICAL_BIAS]
        
        if critical_metrics:
            st.error("🔴 Critical bias detected! Immediate action required:")
            for i, metric in enumerate(critical_metrics, 1):
                st.markdown(f"**{i}.** {metric.recommendation}")
        
        # General recommendations
        st.markdown("### General Recommendations")
        for i, recommendation in enumerate(bias_report.mitigation_recommendations, 1):
            st.markdown(f"**{i}.** {recommendation}")
        
        # Implementation guidance
        st.markdown("### Implementation Guidance")
        
        implementation_steps = [
            "**Immediate (0-7 days):** Address critical bias findings and implement threshold adjustments",
            "**Short-term (1-4 weeks):** Implement post-processing fairness corrections and monitoring",
            "**Medium-term (1-3 months):** Review and retrain models with balanced datasets",
            "**Long-term (3-6 months):** Establish ongoing bias monitoring and automated alerting"
        ]
        
        for step in implementation_steps:
            st.markdown(f"• {step}")
        
        # Export recommendations
        if st.button("📄 Export Recommendations"):
            recommendations_text = self._format_recommendations_for_export(bias_report)
            st.download_button(
                label="📄 Download Recommendations (Text)",
                data=recommendations_text,
                file_name=f"bias_mitigation_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    def _display_detailed_report_tab(self, bias_report: BiasReport):
        """Display detailed technical report"""
        st.subheader("📋 Detailed Technical Report")
        
        # Report metadata
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Analysis Timestamp:** {bias_report.timestamp}")
            st.markdown(f"**Dataset Size:** {bias_report.dataset_size:,}")
            st.markdown(f"**Confidence Score:** {bias_report.confidence_score:.1%}")
        
        with col2:
            st.markdown(f"**Overall Fairness:** {bias_report.overall_fairness.value}")
            st.markdown(f"**Statistical Significance:** {'Yes' if bias_report.statistical_significance else 'No'}")
            st.markdown(f"**Metrics Analyzed:** {len(bias_report.bias_metrics)}")
        
        # Segments analyzed
        st.markdown("**Segments Analyzed:**")
        st.write(", ".join(bias_report.segments_analyzed))
        
        # Full report export
        if st.button("📄 Export Full Report"):
            report_json = self._format_report_for_export(bias_report)
            st.download_button(
                label="📄 Download Full Report (JSON)",
                data=report_json,
                file_name=f"bias_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Raw report data
        with st.expander("Raw Report Data"):
            # Convert report to dict for display
            report_dict = {
                'timestamp': bias_report.timestamp,
                'dataset_size': bias_report.dataset_size,
                'segments_analyzed': bias_report.segments_analyzed,
                'overall_fairness': bias_report.overall_fairness.value,
                'confidence_score': bias_report.confidence_score,
                'statistical_significance': bias_report.statistical_significance,
                'mitigation_recommendations': bias_report.mitigation_recommendations,
                'bias_metrics': [
                    {
                        'metric_name': m.metric_name,
                        'bias_type': m.bias_type.value,
                        'value': m.value,
                        'threshold': m.threshold,
                        'is_biased': m.is_biased,
                        'fairness_level': m.fairness_level.value,
                        'segments': m.segments,
                        'recommendation': m.recommendation,
                        'details': m.details
                    } for m in bias_report.bias_metrics
                ]
            }
            st.json(report_dict)
    
    def _format_recommendations_for_export(self, bias_report: BiasReport) -> str:
        """Format recommendations for text export"""
        lines = [
            "BIAS MITIGATION RECOMMENDATIONS",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Fairness Level: {bias_report.overall_fairness.value}",
            f"Dataset Size: {bias_report.dataset_size:,}",
            "",
            "PRIORITY RECOMMENDATIONS:",
            "-" * 25
        ]
        
        for i, rec in enumerate(bias_report.mitigation_recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        lines.extend([
            "",
            "SPECIFIC METRIC RECOMMENDATIONS:",
            "-" * 35
        ])
        
        for metric in bias_report.bias_metrics:
            if metric.is_biased:
                lines.extend([
                    f"• {metric.metric_name}:",
                    f"  Status: {metric.fairness_level.value}",
                    f"  Action: {metric.recommendation}",
                    ""
                ])
        
        return "\n".join(lines)
    
    def _format_report_for_export(self, bias_report: BiasReport) -> str:
        """Format full report for JSON export"""
        report_dict = {
            'metadata': {
                'timestamp': bias_report.timestamp,
                'dataset_size': bias_report.dataset_size,
                'confidence_score': bias_report.confidence_score,
                'statistical_significance': bias_report.statistical_significance
            },
            'assessment': {
                'overall_fairness': bias_report.overall_fairness.value,
                'segments_analyzed': bias_report.segments_analyzed,
                'mitigation_recommendations': bias_report.mitigation_recommendations
            },
            'metrics': [
                {
                    'name': m.metric_name,
                    'type': m.bias_type.value,
                    'value': m.value,
                    'threshold': m.threshold,
                    'is_biased': m.is_biased,
                    'fairness_level': m.fairness_level.value,
                    'segments': m.segments,
                    'recommendation': m.recommendation,
                    'technical_details': m.details
                } for m in bias_report.bias_metrics
            ]
        }
        
        return json.dumps(report_dict, indent=2, default=str)

# Global bias dashboard instance
bias_dashboard = BiasMonitoringDashboard()