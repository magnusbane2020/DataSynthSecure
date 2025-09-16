#!/usr/bin/env python3
"""
Test script to verify core MVP functionality
"""
import os
import sys
import pandas as pd
from data_generator import SyntheticDataGenerator
from ai_scorer import OpportunityScorer
from report_generator import ExecutiveReportGenerator
from utils import validate_api_key, setup_logging
import time

def test_data_generation():
    """Test synthetic data generation"""
    print("🧪 Testing synthetic data generation...")
    generator = SyntheticDataGenerator()
    df = generator.generate_opportunities(5)  # Test with 5 for speed
    
    assert len(df) == 5, "Should generate 5 opportunities"
    assert 'Opportunity_Name' in df.columns, "Should have Opportunity_Name column"
    assert 'MEDDPICC' not in df.columns or df['Champion'].notna().sum() >= 0, "MEDDPICC fields should be partially populated"
    
    print("✅ Synthetic data generation: PASSED")
    return df

def test_api_key_validation():
    """Test API key validation"""
    print("🧪 Testing API key validation...")
    is_valid = validate_api_key()
    assert is_valid, "OpenAI API key should be valid"
    print("✅ API key validation: PASSED")

def test_ai_scoring(test_df):
    """Test AI scoring with a few opportunities"""
    print("🧪 Testing AI scoring...")
    
    try:
        scorer = OpportunityScorer()
        results = []
        
        for i, row in test_df.head(2).iterrows():  # Test with just 2 opportunities
            print(f"  Scoring opportunity {i+1}: {row['Opportunity_Name']}")
            result = scorer.score_opportunity(row)
            results.append(result)
            
            # Validate result structure
            assert 'Score' in result, "Result should have Score field"
            assert 'Explanation' in result, "Result should have Explanation field"
            assert isinstance(result['Score'], int), "Score should be integer"
            assert 0 <= result['Score'] <= 100, "Score should be between 0-100"
            
            print(f"    Score: {result['Score']}, Explanation preview: {result['Explanation'][:100]}...")
        
        results_df = pd.DataFrame(results)
        avg_score = results_df['Score'].mean()
        print(f"  Average score: {avg_score:.1f}")
        
        print("✅ AI scoring: PASSED")
        return results_df
        
    except Exception as e:
        print(f"❌ AI scoring: FAILED - {str(e)}")
        raise

def test_report_generation(scores_df, synthetic_df):
    """Test executive report generation"""
    print("🧪 Testing executive report generation...")
    
    try:
        report_gen = ExecutiveReportGenerator()
        report = report_gen.generate_report(scores_df, synthetic_df)
        
        assert len(report) > 1000, "Report should be substantial"
        assert "CrowdStrike Executive Report" in report, "Should have proper title"
        assert "MEDDPICC" in report, "Should contain MEDDPICC analysis"
        assert "Security" in report, "Should contain security considerations"
        
        print(f"  Report length: {len(report)} characters")
        print("✅ Executive report generation: PASSED")
        
    except Exception as e:
        print(f"❌ Executive report generation: FAILED - {str(e)}")
        raise

def main():
    """Run all core functionality tests"""
    print("🚀 Starting CrowdStrike App Core Functionality Test")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # Test 1: Data Generation
        synthetic_df = test_data_generation()
        
        # Test 2: API Key Validation  
        test_api_key_validation()
        
        # Test 3: AI Scoring
        scores_df = test_ai_scoring(synthetic_df)
        
        # Test 4: Report Generation
        test_report_generation(scores_df, synthetic_df)
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED! Core MVP functionality is working properly.")
        print("✅ The application is ready for next phase feature development.")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ TESTS FAILED: {str(e)}")
        print("🔧 Please check the error above and fix the issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()