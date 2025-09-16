"""
Advanced Bias Detection and Mitigation System for AI Scoring
Provides fairness validation, explainability metrics, and bias monitoring
for enterprise AI ethics and compliance requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.metrics import classification_report
from collections import defaultdict
import json
import time
from datetime import datetime

# Audit trail integration
from audit_trail import log_user_action, log_system_error, EventType, SeverityLevel

logger = logging.getLogger(__name__)

class BiasType(Enum):
    """Types of bias that can be detected"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    SCORE_DISTRIBUTION = "score_distribution"
    THRESHOLD_BIAS = "threshold_bias"
    CORRELATION_BIAS = "correlation_bias"
    TEMPORAL_BIAS = "temporal_bias"

class FairnessLevel(Enum):
    """Fairness assessment levels"""
    FAIR = "fair"
    ATTENTION_NEEDED = "attention_needed"
    BIAS_DETECTED = "bias_detected"
    CRITICAL_BIAS = "critical_bias"

@dataclass
class BiasMetric:
    """Individual bias metric result"""
    metric_name: str
    bias_type: BiasType
    value: float
    threshold: float
    is_biased: bool
    fairness_level: FairnessLevel
    details: Dict[str, Any]
    segments: List[str]
    recommendation: str

@dataclass
class BiasReport:
    """Comprehensive bias analysis report"""
    timestamp: str
    dataset_size: int
    segments_analyzed: List[str]
    overall_fairness: FairnessLevel
    bias_metrics: List[BiasMetric]
    mitigation_recommendations: List[str]
    confidence_score: float
    statistical_significance: bool

class BiasDetector:
    """Advanced bias detection and fairness validation system"""
    
    def __init__(self):
        # Fairness thresholds (configurable for different compliance requirements)
        self.fairness_thresholds = {
            'demographic_parity': 0.1,  # 10% difference threshold
            'score_distribution': 0.15,  # 15% difference in mean scores
            'correlation_threshold': 0.3,  # Significant correlation threshold
            'statistical_significance': 0.05  # p-value threshold
        }
        
        # Segment definitions for bias analysis
        self.bias_segments = {
            'company_size': ['Enterprise', 'Mid-Market', 'SMB'],
            'industry': ['Technology', 'Financial', 'Healthcare', 'Manufacturing', 'Other'],
            'region': ['North America', 'Europe', 'Asia Pacific', 'Latin America'],
            'deal_size': ['Small (<$50K)', 'Medium ($50K-$500K)', 'Large (>$500K)'],
            'sales_stage': ['Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
        }
        
        logger.info("Bias detector initialized with enterprise fairness thresholds")
    
    def detect_bias(self, scored_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> BiasReport:
        """Comprehensive bias detection across multiple dimensions"""
        try:
            # Log bias detection start
            log_user_action("BIAS_DETECTION_STARTED", 
                          details={"dataset_size": len(scored_data)},
                          compliance_tags=["AI_ETHICS", "FAIRNESS"])
            
            # Merge datasets for analysis
            analysis_df = self._prepare_analysis_dataset(scored_data, synthetic_data)
            
            if analysis_df.empty:
                raise ValueError("No data available for bias analysis")
            
            # Generate segments for analysis
            analysis_df = self._generate_analysis_segments(analysis_df)
            
            # Run bias detection tests
            bias_metrics = []
            
            # 1. Demographic parity analysis
            bias_metrics.extend(self._analyze_demographic_parity(analysis_df))
            
            # 2. Score distribution analysis
            bias_metrics.extend(self._analyze_score_distributions(analysis_df))
            
            # 3. Correlation bias analysis
            bias_metrics.extend(self._analyze_correlation_bias(analysis_df))
            
            # 4. Threshold bias analysis
            bias_metrics.extend(self._analyze_threshold_bias(analysis_df))
            
            # 5. Temporal bias analysis
            bias_metrics.extend(self._analyze_temporal_bias(analysis_df))
            
            # Calculate overall fairness assessment
            overall_fairness = self._calculate_overall_fairness(bias_metrics)
            
            # Generate mitigation recommendations
            recommendations = self._generate_mitigation_recommendations(bias_metrics)
            
            # Calculate confidence and significance
            confidence_score = self._calculate_confidence_score(analysis_df, bias_metrics)
            statistical_significance = self._check_statistical_significance(bias_metrics)
            
            # Create bias report
            report = BiasReport(
                timestamp=datetime.utcnow().isoformat(),
                dataset_size=len(analysis_df),
                segments_analyzed=list(analysis_df.columns[analysis_df.columns.str.contains('_segment')]),
                overall_fairness=overall_fairness,
                bias_metrics=bias_metrics,
                mitigation_recommendations=recommendations,
                confidence_score=confidence_score,
                statistical_significance=statistical_significance
            )
            
            # Log bias detection completion
            log_user_action("BIAS_DETECTION_COMPLETED",
                          details={
                              "overall_fairness": overall_fairness.value,
                              "bias_metrics_count": len(bias_metrics),
                              "dataset_size": len(analysis_df),
                              "confidence_score": confidence_score
                          },
                          compliance_tags=["AI_ETHICS", "FAIRNESS"])
            
            # Log critical bias if detected
            if overall_fairness in [FairnessLevel.BIAS_DETECTED, FairnessLevel.CRITICAL_BIAS]:
                log_user_action("CRITICAL_BIAS_DETECTED",
                              severity=SeverityLevel.HIGH,
                              details={"fairness_level": overall_fairness.value, "bias_count": len(bias_metrics)},
                              compliance_tags=["AI_ETHICS", "BIAS_ALERT"])
            
            return report
            
        except Exception as e:
            log_system_error("BIAS_DETECTION_FAILED", {"error": str(e)})
            logger.error(f"Bias detection failed: {str(e)}")
            raise
    
    def _prepare_analysis_dataset(self, scored_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare merged dataset for bias analysis"""
        if 'Opportunity_ID' not in scored_data.columns or 'Opportunity_ID' not in synthetic_data.columns:
            raise ValueError("Both datasets must contain 'Opportunity_ID' column for merging")
        
        # Merge datasets
        analysis_df = scored_data.merge(synthetic_data, on='Opportunity_ID', how='inner')
        
        # Ensure required columns exist
        required_columns = ['Score', 'Amount', 'Account', 'Stage', 'Product']
        missing_columns = [col for col in required_columns if col not in analysis_df.columns]
        if missing_columns:
            logger.warning(f"Missing columns for bias analysis: {missing_columns}")
        
        return analysis_df
    
    def _generate_analysis_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate analysis segments based on opportunity characteristics"""
        analysis_df = df.copy()
        
        # Company size segment (based on deal amount)
        if 'Amount' in analysis_df.columns:
            analysis_df['company_size_segment'] = pd.cut(
                analysis_df['Amount'],
                bins=[0, 50000, 500000, float('inf')],
                labels=['SMB', 'Mid-Market', 'Enterprise'],
                include_lowest=True
            ).astype(str)
        
        # Deal size segment
        if 'Amount' in analysis_df.columns:
            analysis_df['deal_size_segment'] = pd.cut(
                analysis_df['Amount'],
                bins=[0, 50000, 500000, float('inf')],
                labels=['Small (<$50K)', 'Medium ($50K-$500K)', 'Large (>$500K)'],
                include_lowest=True
            ).astype(str)
        
        # Sales stage segment
        if 'Stage' in analysis_df.columns:
            analysis_df['sales_stage_segment'] = analysis_df['Stage'].astype(str)
        
        # Product segment
        if 'Product' in analysis_df.columns:
            analysis_df['product_segment'] = analysis_df['Product'].astype(str)
        
        # Score tier segment
        if 'Score' in analysis_df.columns:
            analysis_df['score_tier_segment'] = pd.cut(
                analysis_df['Score'],
                bins=[0, 30, 60, 100],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            ).astype(str)
        
        return analysis_df
    
    def _analyze_demographic_parity(self, df: pd.DataFrame) -> List[BiasMetric]:
        """Analyze demographic parity across segments"""
        metrics = []
        
        segment_columns = [col for col in df.columns if col.endswith('_segment')]
        
        for segment_col in segment_columns:
            if segment_col not in df.columns:
                continue
                
            segments = df[segment_col].unique()
            if len(segments) < 2:
                continue
            
            # Calculate acceptance rates (high scores as positive outcomes)
            high_score_threshold = 70  # Configurable threshold
            acceptance_rates = {}
            
            for segment in segments:
                segment_data = df[df[segment_col] == segment]
                if len(segment_data) == 0:
                    continue
                acceptance_rates[segment] = (segment_data['Score'] >= high_score_threshold).mean()
            
            if len(acceptance_rates) < 2:
                continue
            
            # Calculate parity difference
            max_rate = max(acceptance_rates.values())
            min_rate = min(acceptance_rates.values())
            parity_difference = max_rate - min_rate
            
            # Determine bias level
            is_biased = parity_difference > self.fairness_thresholds['demographic_parity']
            fairness_level = self._determine_fairness_level(parity_difference, 'demographic_parity')
            
            # Generate recommendation
            recommendation = self._generate_parity_recommendation(segment_col, acceptance_rates, parity_difference)
            
            metric = BiasMetric(
                metric_name=f"Demographic Parity - {segment_col}",
                bias_type=BiasType.DEMOGRAPHIC_PARITY,
                value=parity_difference,
                threshold=self.fairness_thresholds['demographic_parity'],
                is_biased=is_biased,
                fairness_level=fairness_level,
                details={
                    'acceptance_rates': acceptance_rates,
                    'high_score_threshold': high_score_threshold,
                    'segment_counts': df[segment_col].value_counts().to_dict()
                },
                segments=list(segments),
                recommendation=recommendation
            )
            
            metrics.append(metric)
        
        return metrics
    
    def _analyze_score_distributions(self, df: pd.DataFrame) -> List[BiasMetric]:
        """Analyze score distribution differences across segments"""
        metrics = []
        
        segment_columns = [col for col in df.columns if col.endswith('_segment')]
        
        for segment_col in segment_columns:
            if segment_col not in df.columns:
                continue
                
            segments = df[segment_col].unique()
            if len(segments) < 2:
                continue
            
            # Calculate mean scores for each segment
            segment_means = {}
            segment_stds = {}
            
            for segment in segments:
                segment_data = df[df[segment_col] == segment]['Score']
                if len(segment_data) == 0:
                    continue
                segment_means[segment] = segment_data.mean()
                segment_stds[segment] = segment_data.std()
            
            if len(segment_means) < 2:
                continue
            
            # Calculate distribution difference
            max_mean = max(segment_means.values())
            min_mean = min(segment_means.values())
            distribution_difference = (max_mean - min_mean) / max_mean  # Relative difference
            
            # Statistical significance test (ANOVA)
            segment_groups = [df[df[segment_col] == segment]['Score'].values for segment in segments]
            try:
                f_stat, p_value = stats.f_oneway(*segment_groups)
                statistically_significant = p_value < self.fairness_thresholds['statistical_significance']
            except:
                statistically_significant = False
                p_value = 1.0
            
            # Determine bias level
            is_biased = distribution_difference > self.fairness_thresholds['score_distribution'] and statistically_significant
            fairness_level = self._determine_fairness_level(distribution_difference, 'score_distribution')
            
            # Generate recommendation
            recommendation = self._generate_distribution_recommendation(segment_col, segment_means, distribution_difference)
            
            metric = BiasMetric(
                metric_name=f"Score Distribution - {segment_col}",
                bias_type=BiasType.SCORE_DISTRIBUTION,
                value=distribution_difference,
                threshold=self.fairness_thresholds['score_distribution'],
                is_biased=is_biased,
                fairness_level=fairness_level,
                details={
                    'segment_means': segment_means,
                    'segment_stds': segment_stds,
                    'p_value': p_value,
                    'statistically_significant': statistically_significant,
                    'f_statistic': f_stat if 'f_stat' in locals() else None
                },
                segments=list(segments),
                recommendation=recommendation
            )
            
            metrics.append(metric)
        
        return metrics
    
    def _analyze_correlation_bias(self, df: pd.DataFrame) -> List[BiasMetric]:
        """Analyze correlations between scores and protected attributes"""
        metrics = []
        
        if 'Score' not in df.columns:
            return metrics
        
        # Analyze correlations with numerical proxies
        numerical_columns = ['Amount']
        categorical_mappings = {}
        
        # Create numerical mappings for categorical segments
        segment_columns = [col for col in df.columns if col.endswith('_segment')]
        for segment_col in segment_columns:
            if segment_col in df.columns:
                # Create ordinal encoding for correlation analysis
                unique_values = df[segment_col].unique()
                mapping = {val: i for i, val in enumerate(unique_values)}
                categorical_mappings[segment_col] = mapping
                df[f'{segment_col}_numeric'] = df[segment_col].map(mapping)
                numerical_columns.append(f'{segment_col}_numeric')
        
        # Calculate correlations
        for col in numerical_columns:
            if col not in df.columns:
                continue
            
            try:
                correlation = df['Score'].corr(df[col])
                abs_correlation = abs(correlation)
                
                # Determine if correlation indicates bias
                is_biased = abs_correlation > self.fairness_thresholds['correlation_threshold']
                
                if is_biased:
                    fairness_level = FairnessLevel.BIAS_DETECTED if abs_correlation < 0.5 else FairnessLevel.CRITICAL_BIAS
                else:
                    fairness_level = FairnessLevel.FAIR
                
                # Generate recommendation
                if col.endswith('_numeric'):
                    original_col = col.replace('_numeric', '')
                    recommendation = f"Strong correlation detected between scores and {original_col}. Consider feature importance analysis and bias mitigation."
                else:
                    recommendation = f"Correlation between scores and {col} requires monitoring. Consider threshold adjustments."
                
                metric = BiasMetric(
                    metric_name=f"Correlation Bias - {col}",
                    bias_type=BiasType.CORRELATION_BIAS,
                    value=abs_correlation,
                    threshold=self.fairness_thresholds['correlation_threshold'],
                    is_biased=is_biased,
                    fairness_level=fairness_level,
                    details={
                        'correlation': correlation,
                        'direction': 'positive' if correlation > 0 else 'negative',
                        'mapping': categorical_mappings.get(col.replace('_numeric', ''), None)
                    },
                    segments=[col],
                    recommendation=recommendation
                )
                
                metrics.append(metric)
                
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {col}: {str(e)}")
        
        return metrics
    
    def _analyze_threshold_bias(self, df: pd.DataFrame) -> List[BiasMetric]:
        """Analyze bias in threshold-based decisions across segments"""
        metrics = []
        
        segment_columns = [col for col in df.columns if col.endswith('_segment')]
        thresholds = [50, 60, 70, 80]  # Different threshold levels to test
        
        for threshold in thresholds:
            for segment_col in segment_columns:
                if segment_col not in df.columns:
                    continue
                
                segments = df[segment_col].unique()
                if len(segments) < 2:
                    continue
                
                # Calculate threshold-based outcomes for each segment
                segment_outcomes = {}
                for segment in segments:
                    segment_data = df[df[segment_col] == segment]
                    if len(segment_data) == 0:
                        continue
                    segment_outcomes[segment] = (segment_data['Score'] >= threshold).mean()
                
                if len(segment_outcomes) < 2:
                    continue
                
                # Calculate threshold bias
                max_outcome = max(segment_outcomes.values())
                min_outcome = min(segment_outcomes.values())
                threshold_bias = max_outcome - min_outcome
                
                # Determine bias level
                is_biased = threshold_bias > self.fairness_thresholds['demographic_parity']
                fairness_level = self._determine_fairness_level(threshold_bias, 'demographic_parity')
                
                metric = BiasMetric(
                    metric_name=f"Threshold Bias (T={threshold}) - {segment_col}",
                    bias_type=BiasType.THRESHOLD_BIAS,
                    value=threshold_bias,
                    threshold=self.fairness_thresholds['demographic_parity'],
                    is_biased=is_biased,
                    fairness_level=fairness_level,
                    details={
                        'threshold_used': threshold,
                        'segment_outcomes': segment_outcomes,
                        'segments_analyzed': list(segments)
                    },
                    segments=list(segments),
                    recommendation=f"Consider threshold adjustment or segment-specific calibration for threshold {threshold}"
                )
                
                if is_biased:  # Only report biased thresholds to reduce noise
                    metrics.append(metric)
        
        return metrics
    
    def _analyze_temporal_bias(self, df: pd.DataFrame) -> List[BiasMetric]:
        """Analyze temporal bias in scoring patterns"""
        metrics = []
        
        if 'Scoring_Timestamp' not in df.columns:
            return metrics
        
        try:
            # Convert timestamp to datetime if it's string
            df['timestamp_dt'] = pd.to_datetime(df['Scoring_Timestamp'])
            
            # Group by time periods (e.g., by hour or day)
            df['time_period'] = df['timestamp_dt'].dt.floor('H')  # Group by hour
            
            time_periods = df['time_period'].unique()
            if len(time_periods) < 2:
                return metrics
            
            # Calculate score statistics by time period
            temporal_stats = df.groupby('time_period')['Score'].agg(['mean', 'std', 'count']).reset_index()
            
            # Calculate temporal variance
            temporal_variance = temporal_stats['mean'].var()
            mean_of_means = temporal_stats['mean'].mean()
            temporal_coefficient = temporal_variance / mean_of_means if mean_of_means > 0 else 0
            
            # Determine if temporal bias exists
            is_biased = temporal_coefficient > 0.1  # 10% coefficient of variation threshold
            fairness_level = FairnessLevel.ATTENTION_NEEDED if is_biased else FairnessLevel.FAIR
            
            metric = BiasMetric(
                metric_name="Temporal Bias Analysis",
                bias_type=BiasType.TEMPORAL_BIAS,
                value=temporal_coefficient,
                threshold=0.1,
                is_biased=is_biased,
                fairness_level=fairness_level,
                details={
                    'temporal_variance': temporal_variance,
                    'mean_score': mean_of_means,
                    'time_periods_count': len(time_periods),
                    'temporal_stats': temporal_stats.to_dict('records')
                },
                segments=['temporal'],
                recommendation="Monitor scoring consistency over time. Consider model drift detection and retraining schedules."
            )
            
            metrics.append(metric)
            
        except Exception as e:
            logger.warning(f"Could not perform temporal bias analysis: {str(e)}")
        
        return metrics
    
    def _determine_fairness_level(self, value: float, metric_type: str) -> FairnessLevel:
        """Determine fairness level based on metric value and type"""
        threshold = self.fairness_thresholds.get(metric_type, 0.1)
        
        if value <= threshold:
            return FairnessLevel.FAIR
        elif value <= threshold * 2:
            return FairnessLevel.ATTENTION_NEEDED
        elif value <= threshold * 3:
            return FairnessLevel.BIAS_DETECTED
        else:
            return FairnessLevel.CRITICAL_BIAS
    
    def _calculate_overall_fairness(self, bias_metrics: List[BiasMetric]) -> FairnessLevel:
        """Calculate overall fairness assessment from individual metrics"""
        if not bias_metrics:
            return FairnessLevel.FAIR
        
        # Count metrics by fairness level
        level_counts = defaultdict(int)
        for metric in bias_metrics:
            level_counts[metric.fairness_level] += 1
        
        # Determine overall assessment
        if level_counts[FairnessLevel.CRITICAL_BIAS] > 0:
            return FairnessLevel.CRITICAL_BIAS
        elif level_counts[FairnessLevel.BIAS_DETECTED] >= 2:
            return FairnessLevel.BIAS_DETECTED
        elif level_counts[FairnessLevel.ATTENTION_NEEDED] >= 3:
            return FairnessLevel.ATTENTION_NEEDED
        else:
            return FairnessLevel.FAIR
    
    def _generate_mitigation_recommendations(self, bias_metrics: List[BiasMetric]) -> List[str]:
        """Generate actionable bias mitigation recommendations"""
        recommendations = []
        
        # Generic recommendations based on detected bias types
        bias_types_detected = {metric.bias_type for metric in bias_metrics if metric.is_biased}
        
        if BiasType.DEMOGRAPHIC_PARITY in bias_types_detected:
            recommendations.append("Implement demographic parity constraints in scoring model")
            recommendations.append("Consider post-processing fairness adjustments")
        
        if BiasType.SCORE_DISTRIBUTION in bias_types_detected:
            recommendations.append("Calibrate scoring thresholds by segment")
            recommendations.append("Review training data for representation issues")
        
        if BiasType.CORRELATION_BIAS in bias_types_detected:
            recommendations.append("Remove or transform highly correlated features")
            recommendations.append("Implement feature importance analysis")
        
        if BiasType.THRESHOLD_BIAS in bias_types_detected:
            recommendations.append("Implement segment-specific decision thresholds")
            recommendations.append("Consider adaptive threshold systems")
        
        if BiasType.TEMPORAL_BIAS in bias_types_detected:
            recommendations.append("Implement model drift detection and monitoring")
            recommendations.append("Schedule regular model retraining")
        
        # Add specific recommendations from individual metrics
        for metric in bias_metrics:
            if metric.is_biased and metric.recommendation:
                recommendations.append(metric.recommendation)
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_confidence_score(self, df: pd.DataFrame, bias_metrics: List[BiasMetric]) -> float:
        """Calculate confidence score for bias analysis"""
        # Base confidence on dataset size and statistical significance
        base_confidence = min(0.9, len(df) / 1000)  # Higher confidence with more data
        
        # Adjust for statistical significance
        significant_tests = sum(1 for metric in bias_metrics 
                              if metric.details.get('statistically_significant', False))
        total_tests = len(bias_metrics)
        
        if total_tests > 0:
            significance_factor = significant_tests / total_tests
            final_confidence = base_confidence * (0.5 + 0.5 * significance_factor)
        else:
            final_confidence = base_confidence * 0.5
        
        return round(final_confidence, 2)
    
    def _check_statistical_significance(self, bias_metrics: List[BiasMetric]) -> bool:
        """Check if bias findings are statistically significant"""
        significant_metrics = [metric for metric in bias_metrics 
                             if metric.details.get('statistically_significant', False)]
        return len(significant_metrics) > 0
    
    def _generate_parity_recommendation(self, segment_col: str, acceptance_rates: Dict, difference: float) -> str:
        """Generate specific recommendation for demographic parity issues"""
        worst_segment = min(acceptance_rates.keys(), key=lambda k: acceptance_rates[k])
        best_segment = max(acceptance_rates.keys(), key=lambda k: acceptance_rates[k])
        
        return f"Demographic parity violation in {segment_col}: {worst_segment} segment has {difference:.1%} lower acceptance rate than {best_segment}. Consider bias mitigation techniques."
    
    def _generate_distribution_recommendation(self, segment_col: str, segment_means: Dict, difference: float) -> str:
        """Generate specific recommendation for score distribution issues"""
        lowest_segment = min(segment_means.keys(), key=lambda k: segment_means[k])
        highest_segment = max(segment_means.keys(), key=lambda k: segment_means[k])
        
        return f"Score distribution bias in {segment_col}: {highest_segment} segment scores {difference:.1%} higher than {lowest_segment}. Review feature weighting and training data balance."

# Global bias detector instance
bias_detector = BiasDetector()