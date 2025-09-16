import os
import json
import logging
from openai import OpenAI
import time

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
        logger.info("OpenAI client initialized securely")

    def score_opportunity(self, opportunity_row):
        """Score a single opportunity using AI with MEDDPICC/BANT framework"""
        try:
            # Prepare opportunity data for scoring
            opp_data = self._prepare_opportunity_data(opportunity_row)
            
            # Create scoring prompt
            prompt = self._create_scoring_prompt(opp_data)
            
            # Call OpenAI API
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
            )
            
            # Parse response
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
