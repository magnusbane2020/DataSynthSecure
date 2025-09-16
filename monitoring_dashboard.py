"""
Real-time Monitoring Dashboard for Audit Trail System
Provides live monitoring, anomaly detection, and compliance visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional
import time
import numpy as np

from audit_trail import audit_manager, EventType, SeverityLevel

class MonitoringDashboard:
    """Real-time monitoring dashboard for audit trail system"""
    
    def __init__(self):
        self.audit_manager = audit_manager
        self.color_palette = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'warning': '#ff7f0e', 
            'danger': '#d62728',
            'info': '#17becf',
            'secondary': '#7f7f7f'
        }
    
    def display_monitoring_dashboard(self):
        """Main monitoring dashboard display"""
        st.header("🔒 Security & Audit Monitoring Dashboard")
        st.markdown("Real-time monitoring of audit events, security incidents, and compliance status")
        
        # Dashboard controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_range = st.selectbox(
                "Time Range",
                ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
                index=2
            )
        
        with col2:
            auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Auto refresh using Streamlit's built-in mechanism (non-blocking)
        if auto_refresh:
            st_autorefresh = st.empty()
            with st_autorefresh:
                if st.button("🔄 Auto-refreshing in 30s..."):
                    st.rerun()
        
        # Convert time range to hours
        time_mapping = {
            "Last 1 Hour": 1,
            "Last 6 Hours": 6,
            "Last 24 Hours": 24,
            "Last 7 Days": 168,
            "Last 30 Days": 720
        }
        hours_back = time_mapping[time_range]
        
        # Get audit events for the selected time range
        start_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
        events = self.audit_manager.get_events(start_time=start_time, limit=5000)
        
        if not events:
            st.warning("No audit events found for the selected time period.")
            return
        
        # Convert to DataFrame for analysis with type normalization
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Normalize data types to prevent display errors
        df['event_type'] = df['event_type'].astype(str)
        df['severity'] = df['severity'].astype(str)
        df['outcome'] = df['outcome'].astype(str)
        
        # Parse JSON fields safely
        def safe_json_parse(x):
            if isinstance(x, str):
                try:
                    return json.loads(x)
                except:
                    return {}
            return x if isinstance(x, dict) else {}
        
        df['details'] = df['details'].apply(safe_json_parse)
        df['compliance_tags'] = df['compliance_tags'].apply(lambda x: x if isinstance(x, list) else [])
        
        # Display key metrics
        self.display_key_metrics(df)
        
        # Security alerts section
        self.display_security_alerts(df)
        
        # Visualizations in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Activity Overview", 
            "🚨 Security Events", 
            "👥 User Activity", 
            "📈 Performance Metrics",
            "📋 Compliance Status"
        ])
        
        with tab1:
            self.display_activity_overview(df)
        
        with tab2:
            self.display_security_analysis(df)
        
        with tab3:
            self.display_user_activity(df)
        
        with tab4:
            self.display_performance_metrics(df)
        
        with tab5:
            self.display_compliance_status(df)
    
    def display_key_metrics(self, df: pd.DataFrame):
        """Display key monitoring metrics"""
        st.subheader("📊 Real-Time Metrics")
        
        # Calculate metrics
        total_events = len(df)
        security_events = len(df[df['event_type'] == 'security_event'])
        failed_events = len(df[df['outcome'] != 'SUCCESS'])
        critical_events = len(df[df['severity'] == 'critical'])
        unique_users = df['user_id'].nunique()
        
        # Success rate
        success_rate = ((total_events - failed_events) / total_events * 100) if total_events > 0 else 100
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Events",
                f"{total_events:,}",
                delta=f"+{len(df[df['timestamp'] > datetime.utcnow() - timedelta(hours=1)]):,} (1h)"
            )
        
        with col2:
            delta_color = "inverse" if security_events > 0 else "normal"
            st.metric(
                "Security Events", 
                f"{security_events:,}",
                delta=f"{security_events} active",
                delta_color=delta_color
            )
        
        with col3:
            delta_color = "inverse" if failed_events > 0 else "normal"
            st.metric(
                "Failed Events",
                f"{failed_events:,}",
                delta=f"{(failed_events/total_events*100):.1f}% rate" if total_events > 0 else "0%",
                delta_color=delta_color
            )
        
        with col4:
            delta_color = "inverse" if critical_events > 0 else "normal"
            st.metric(
                "Critical Events",
                f"{critical_events:,}",
                delta=f"Severity: Critical" if critical_events > 0 else "All Clear",
                delta_color=delta_color
            )
        
        with col5:
            st.metric(
                "Active Users",
                f"{unique_users:,}",
                delta=f"Success: {success_rate:.1f}%"
            )
    
    def display_security_alerts(self, df: pd.DataFrame):
        """Display active security alerts"""
        # Filter for security and critical events
        security_df = df[
            (df['event_type'] == 'security_event') | 
            (df['severity'] == 'critical') |
            (df['outcome'] != 'SUCCESS')
        ].sort_values('timestamp', ascending=False)
        
        if not security_df.empty:
            st.subheader("🚨 Active Security Alerts")
            
            for _, event in security_df.head(5).iterrows():
                severity_color = {
                    'critical': 'red',
                    'high': 'orange',
                    'medium': 'yellow',
                    'low': 'blue'
                }.get(event['severity'], 'gray')
                
                with st.expander(f"🚨 {event['action']} - {event['severity'].upper()}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Event ID:** {event['event_id']}")
                        st.write(f"**Timestamp:** {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
                        st.write(f"**User:** {event['user_id']}")
                        st.write(f"**Outcome:** {event['outcome']}")
                    
                    with col2:
                        st.write(f"**Event Type:** {event['event_type']}")
                        st.write(f"**Resource:** {event.get('resource', 'N/A')}")
                        st.write(f"**IP Address:** {event.get('ip_address', 'N/A')}")
                        
                        if event['details']:
                            with st.expander("📋 Event Details"):
                                st.json(event['details'])
    
    def display_activity_overview(self, df: pd.DataFrame):
        """Display activity overview charts"""
        st.subheader("📊 Activity Overview")
        
        # Time series chart of events
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_events = df.groupby('hour').size().reset_index(name='count')
        
        fig_timeline = px.line(
            hourly_events, 
            x='hour', 
            y='count',
            title="Event Volume Over Time",
            labels={'count': 'Number of Events', 'hour': 'Time'}
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Event type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            event_type_counts = df['event_type'].value_counts()
            fig_types = px.pie(
                values=event_type_counts.values,
                names=event_type_counts.index,
                title="Event Types Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            severity_counts = df['severity'].value_counts()
            colors = ['red', 'orange', 'yellow', 'blue'][:len(severity_counts)]
            fig_severity = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                title="Events by Severity Level",
                color=severity_counts.index,
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig_severity, use_container_width=True)
    
    def display_security_analysis(self, df: pd.DataFrame):
        """Display security-focused analysis"""
        st.subheader("🚨 Security Event Analysis")
        
        # Security events over time
        security_df = df[df['event_type'] == 'security_event']
        
        if security_df.empty:
            st.success("✅ No security events detected in the selected time period.")
            return
        
        # Security events timeline
        security_df['hour'] = security_df['timestamp'].dt.floor('H')
        security_hourly = security_df.groupby(['hour', 'severity']).size().reset_index(name='count')
        
        fig_security = px.bar(
            security_hourly,
            x='hour',
            y='count',
            color='severity',
            title="Security Events Timeline by Severity",
            color_discrete_map={
                'critical': '#d62728',
                'high': '#ff7f0e',
                'medium': '#2ca02c',
                'low': '#1f77b4'
            }
        )
        st.plotly_chart(fig_security, use_container_width=True)
        
        # Top security event types
        security_actions = security_df['action'].value_counts().head(10)
        fig_actions = px.bar(
            x=security_actions.values,
            y=security_actions.index,
            orientation='h',
            title="Top Security Event Types",
            labels={'x': 'Count', 'y': 'Event Type'}
        )
        st.plotly_chart(fig_actions, use_container_width=True)
        
        # Security events table
        st.subheader("Recent Security Events")
        security_display = security_df[['timestamp', 'action', 'user_id', 'severity', 'outcome']].head(10)
        st.dataframe(security_display, use_container_width=True)
    
    def display_user_activity(self, df: pd.DataFrame):
        """Display user activity analysis"""
        st.subheader("👥 User Activity Analysis")
        
        # User activity metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Top users by activity
            user_activity = df['user_id'].value_counts().head(10)
            fig_users = px.bar(
                x=user_activity.values,
                y=user_activity.index,
                orientation='h',
                title="Most Active Users",
                labels={'x': 'Number of Events', 'y': 'User ID'}
            )
            st.plotly_chart(fig_users, use_container_width=True)
        
        with col2:
            # User activity by hour
            df['hour_of_day'] = df['timestamp'].dt.hour
            hourly_user_activity = df.groupby('hour_of_day')['user_id'].nunique().reset_index()
            hourly_user_activity.columns = ['hour', 'unique_users']
            
            fig_hourly = px.line(
                hourly_user_activity,
                x='hour',
                y='unique_users',
                title="Unique Users by Hour of Day",
                labels={'unique_users': 'Unique Users', 'hour': 'Hour of Day'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Anomaly detection for users
        self.display_user_anomalies(df)
        
        # Detailed user table
        st.subheader("User Activity Summary")
        user_summary = df.groupby('user_id').agg({
            'event_id': 'count',
            'timestamp': ['min', 'max'],
            'outcome': lambda x: (x != 'SUCCESS').sum(),
            'severity': lambda x: (x == 'critical').sum()
        }).round(2)
        
        user_summary.columns = ['Total Events', 'First Activity', 'Last Activity', 'Failed Events', 'Critical Events']
        st.dataframe(user_summary, use_container_width=True)
    
    def display_user_anomalies(self, df: pd.DataFrame):
        """Detect and display user behavior anomalies"""
        st.subheader("🔍 Anomaly Detection")
        
        # Calculate user activity statistics
        user_stats = df.groupby('user_id').agg({
            'event_id': 'count',
            'timestamp': lambda x: (x.max() - x.min()).total_seconds() / 3600,  # Duration in hours
            'outcome': lambda x: (x != 'SUCCESS').sum() / len(x) * 100,  # Failure rate
            'severity': lambda x: (x.isin(['critical', 'high'])).sum()  # High severity events
        }).round(2)
        
        user_stats.columns = ['total_events', 'session_hours', 'failure_rate', 'high_severity_events']
        
        # Detect anomalies using statistical thresholds
        anomalies = []
        
        # High activity anomaly (>3 standard deviations)
        activity_threshold = user_stats['total_events'].mean() + 3 * user_stats['total_events'].std()
        high_activity_users = user_stats[user_stats['total_events'] > activity_threshold]
        
        for user_id in high_activity_users.index:
            anomalies.append({
                'user_id': user_id,
                'anomaly_type': 'HIGH_ACTIVITY',
                'severity': 'MEDIUM',
                'details': f"{high_activity_users.loc[user_id, 'total_events']} events (threshold: {activity_threshold:.0f})"
            })
        
        # High failure rate anomaly (>20%)
        high_failure_users = user_stats[user_stats['failure_rate'] > 20]
        for user_id in high_failure_users.index:
            anomalies.append({
                'user_id': user_id,
                'anomaly_type': 'HIGH_FAILURE_RATE',
                'severity': 'HIGH',
                'details': f"{high_failure_users.loc[user_id, 'failure_rate']:.1f}% failure rate"
            })
        
        # Multiple critical events
        critical_users = user_stats[user_stats['high_severity_events'] > 0]
        for user_id in critical_users.index:
            anomalies.append({
                'user_id': user_id,
                'anomaly_type': 'CRITICAL_EVENTS',
                'severity': 'CRITICAL',
                'details': f"{critical_users.loc[user_id, 'high_severity_events']} critical/high severity events"
            })
        
        if anomalies:
            st.warning(f"⚠️ {len(anomalies)} user behavior anomalies detected")
            
            anomaly_df = pd.DataFrame(anomalies)
            st.dataframe(anomaly_df, use_container_width=True)
        else:
            st.success("✅ No user behavior anomalies detected")
    
    def display_performance_metrics(self, df: pd.DataFrame):
        """Display system performance metrics"""
        st.subheader("📈 Performance Metrics")
        
        # API call performance
        api_events = df[df['event_type'] == 'api_call']
        
        if not api_events.empty and 'duration_ms' in api_events.columns:
            # Response time analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Response time distribution
                api_events_with_duration = api_events.dropna(subset=['duration_ms'])
                if not api_events_with_duration.empty:
                    fig_response = px.histogram(
                        api_events_with_duration,
                        x='duration_ms',
                        title="API Response Time Distribution",
                        labels={'duration_ms': 'Response Time (ms)', 'count': 'Number of Calls'}
                    )
                    st.plotly_chart(fig_response, use_container_width=True)
            
            with col2:
                # Performance by endpoint
                if 'resource' in api_events.columns:
                    endpoint_performance = api_events.groupby('resource')['duration_ms'].agg(['mean', 'count']).reset_index()
                    endpoint_performance.columns = ['endpoint', 'avg_response_time', 'call_count']
                    endpoint_performance = endpoint_performance.sort_values('avg_response_time', ascending=False)
                    
                    fig_endpoints = px.bar(
                        endpoint_performance.head(10),
                        x='avg_response_time',
                        y='endpoint',
                        orientation='h',
                        title="Average Response Time by Endpoint",
                        labels={'avg_response_time': 'Average Response Time (ms)', 'endpoint': 'API Endpoint'}
                    )
                    st.plotly_chart(fig_endpoints, use_container_width=True)
        
        # Success rate over time
        df['hour'] = df['timestamp'].dt.floor('H')
        success_rate = df.groupby('hour').apply(
            lambda x: (x['outcome'] == 'SUCCESS').sum() / len(x) * 100
        ).reset_index(name='success_rate')
        
        fig_success = px.line(
            success_rate,
            x='hour',
            y='success_rate',
            title="Success Rate Over Time",
            labels={'success_rate': 'Success Rate (%)', 'hour': 'Time'}
        )
        fig_success.add_hline(y=95, line_dash="dash", line_color="red", 
                             annotation_text="95% SLA Threshold")
        st.plotly_chart(fig_success, use_container_width=True)
    
    def display_compliance_status(self, df: pd.DataFrame):
        """Display compliance monitoring status"""
        st.subheader("📋 Compliance Status")
        
        # Generate compliance report
        report = self.audit_manager.generate_compliance_report(days_back=1)
        
        # Compliance score overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sox_status = report['compliance_status']['SOX']
            sox_color = "success" if sox_status == "COMPLIANT" else "error"
            st.metric("SOX Compliance", sox_status, delta="Audit Trail Integrity")
            if sox_status != "COMPLIANT":
                st.error("⚠️ SOX compliance requires attention")
        
        with col2:
            gdpr_status = report['compliance_status']['GDPR']
            gdpr_color = "success" if gdpr_status == "COMPLIANT" else "error"
            st.metric("GDPR Compliance", gdpr_status, delta="Data Protection")
            if gdpr_status != "COMPLIANT":
                st.error("⚠️ GDPR compliance requires review")
        
        with col3:
            hipaa_status = report['compliance_status']['HIPAA']
            hipaa_color = "success" if hipaa_status == "COMPLIANT" else "error"
            st.metric("HIPAA Compliance", hipaa_status, delta="Security Standards")
            if hipaa_status != "COMPLIANT":
                st.error("⚠️ HIPAA compliance requires attention")
        
        # Integrity verification results
        st.subheader("🔒 Audit Trail Integrity")
        integrity_result = report['integrity_verification']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Integrity Score", f"{integrity_result['integrity_score']:.1f}%")
        
        with col2:
            st.metric("Verified Events", f"{integrity_result['verified_events']:,}")
        
        with col3:
            tampered_count = integrity_result['tampered_events']
            delta_color = "inverse" if tampered_count > 0 else "normal"
            st.metric("Tampered Events", f"{tampered_count:,}", delta_color=delta_color)
        
        if integrity_result['tampered_events'] > 0:
            st.error("🚨 CRITICAL: Tampered audit events detected! Immediate investigation required.")
            with st.expander("Tampered Events Details"):
                st.json(integrity_result['tampered_event_details'])
        else:
            st.success("✅ Audit trail integrity verified - No tampering detected")
        
        # Compliance report download
        if st.button("📄 Download Compliance Report"):
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="📄 Download Report (JSON)",
                data=report_json,
                file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Global monitoring dashboard instance
monitoring_dashboard = MonitoringDashboard()