import os
import json
import logging
from openai import OpenAI
import time
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import random

logger = logging.getLogger(__name__)

class OpportunityScorer:
    """Secure AI-powered opportunity scorer using OpenAI"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-5"
        
        # Secure API key handling
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        
        self.client = OpenAI(api_key=api_key)
        
        # Batch processing configuration
        self.batch_size = 10  # Process 10 opportunities at a time
        self.max_retries = 3
        self.base_delay = 1  # Base delay for exponential backoff
        self.timeout = 30  # Request timeout in seconds
        
        logger.info("OpenAI client initialized securely with batch processing capabilities")

    def score_opportunity(self, opportunity_row):
        """Score a single opportunity using AI with MEDDPICC/BANT framework"""
        try:
            # Prepare opportunity data for scoring
            opp_data = self._prepare_opportunity_data(opportunity_row)
            
            # Create scoring prompt
            prompt = self._create_scoring_prompt(opp_data)
            
            # Call OpenAI API with timeout and retry logic
            response = self._make_api_call_with_retry(prompt)
            
            # Parse response
            if not response or not response.choices:
                raise ValueError("Invalid response from OpenAI API")
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")
            result = json.loads(content)
            
            # Validate and format result
            score = max(0, min(100, int(result.get('score', 0))))
            explanation = result.get('explanation', 'No explanation provided')
            
            return {
                'Opportunity_ID': opportunity_row.get('Opportunity_ID', 'Unknown'),
                'Opportunity_Name': opportunity_row.get('Opportunity_Name', 'Unknown'),
                'Score': score,
                'Explanation': explanation,
                'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error scoring opportunity: {str(e)}")
            return {
                'Opportunity_ID': opportunity_row.get('Opportunity_ID', 'Unknown'),
                'Opportunity_Name': opportunity_row.get('Opportunity_Name', 'Unknown'),
                'Score': 0,
                'Explanation': f"Scoring error: {str(e)}",
                'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def _prepare_opportunity_data(self, row):
        """Prepare opportunity data for scoring, removing any sensitive info"""
        return {
            'opportunity_name': row.get('Opportunity_Name', ''),
            'account': row.get('Account', ''),
            'amount': row.get('Amount', 0),
            'stage': row.get('Stage', ''),
            'product': row.get('Product', ''),
            'notes': row.get('Opportunity_Notes', ''),
            'metrics': row.get('Metrics', ''),
            'economic_buyer': row.get('Economic_Buyer', ''),
            'decision_criteria': row.get('Decision_Criteria', ''),
            'decision_process': row.get('Decision_Process', ''),
            'paper_process': row.get('Paper_Process', ''),
            'identify_pain': row.get('Identify_Pain', ''),
            'champion': row.get('Champion', ''),
            'competition': row.get('Competition', '')
        }

    def _create_scoring_prompt(self, opp_data):
        """Create comprehensive scoring prompt"""
        return f"""
Analyze this cybersecurity opportunity and provide a score from 0-100 based on MEDDPICC and BANT criteria.

OPPORTUNITY DETAILS:
- Name: {opp_data['opportunity_name']}
- Account: {opp_data['account']}
- Amount: ${opp_data['amount']:,}
- Stage: {opp_data['stage']}
- Product: {opp_data['product']}
- Notes: {opp_data['notes']}

MEDDPICC ANALYSIS:
- Metrics: {opp_data['metrics'] or 'Not defined'}
- Economic Buyer: {opp_data['economic_buyer'] or 'Not identified'}
- Decision Criteria: {opp_data['decision_criteria'] or 'Not defined'}
- Decision Process: {opp_data['decision_process'] or 'Not defined'}
- Paper Process: {opp_data['paper_process'] or 'Not defined'}
- Identify Pain: {opp_data['identify_pain'] or 'Not identified'}
- Champion: {opp_data['champion'] or 'Not identified'}
- Competition: {opp_data['competition'] or 'Not identified'}

SCORING CRITERIA:
- MEDDPICC Completeness (40 points): How many elements are well-defined?
- Business Value (25 points): Clear ROI, metrics, and pain points?
- Buyer Engagement (20 points): Economic buyer identified, champion present?
- Process Clarity (15 points): Decision process and timeline defined?

Provide your response in JSON format:
{{
    "score": <integer from 0-100>,
    "explanation": "<detailed explanation of score including strengths, weaknesses, and recommendations>"
}}
"""

    def _make_api_call_with_retry(self, prompt: str):
        """Make OpenAI API call with retry logic and exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a cybersecurity sales expert specializing in opportunity qualification using MEDDPICC and BANT frameworks. Analyze opportunities and provide scores with detailed explanations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"}
                    # Note: GPT-5 only supports default temperature of 1.0
                    # Timeout is handled by the client internally
                )
                return response
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} API call attempts failed: {str(e)}")
                    raise

    def score_opportunities_batch(self, opportunities_df, batch_size: Optional[int] = None, progress_callback=None):
        """Score multiple opportunities in batches with progress tracking"""
        if batch_size is None:
            batch_size = self.batch_size
            
        total_opportunities = len(opportunities_df)
        results = []
        failed_count = 0
        
        logger.info(f"Starting batch scoring for {total_opportunities} opportunities (batch size: {batch_size})")
        
        for batch_start in range(0, total_opportunities, batch_size):
            batch_end = min(batch_start + batch_size, total_opportunities)
            batch_num = batch_start // batch_size + 1
            total_batches = (total_opportunities - 1) // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({batch_start+1}-{batch_end} of {total_opportunities})")
            
            batch_opportunities = opportunities_df.iloc[batch_start:batch_end]
            batch_results = []
            
            for idx, row in batch_opportunities.iterrows():
                try:
                    result = self.score_opportunity(row)
                    batch_results.append(result)
                    
                    if progress_callback:
                        progress_callback(batch_start + len(batch_results), total_opportunities)
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to score opportunity {row.get('Opportunity_Name', 'Unknown')}: {str(e)}")
                    # Add failed result
                    batch_results.append({
                        'Opportunity_ID': row.get('Opportunity_ID', 'Unknown'),
                        'Opportunity_Name': row.get('Opportunity_Name', 'Unknown'),
                        'Score': 0,
                        'Explanation': f"Scoring failed: {str(e)}",
                        'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    if progress_callback:
                        progress_callback(batch_start + len(batch_results), total_opportunities)
            
            results.extend(batch_results)
            
            # Add small delay between batches to avoid rate limiting
            if batch_end < total_opportunities:
                time.sleep(0.5)
        
        # Count actual failures by checking results (since score_opportunity swallows errors)
        actual_failures = sum(1 for result in results if result['Score'] == 0 and 'error' in result['Explanation'].lower())
        success_count = total_opportunities - actual_failures
        success_rate = (success_count / total_opportunities) * 100
        
        logger.info(f"Batch scoring completed. Success rate: {success_rate:.1f}% ({success_count}/{total_opportunities})")
        
        return results, actual_failures
