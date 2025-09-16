import os
import json
import logging
from openai import OpenAI
import time
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from database import get_db_manager
import pandas as pd

logger = logging.getLogger(__name__)

class OpportunityScorer:
    """Secure AI-powered opportunity scorer using OpenAI"""
    
    def __init__(self):
        # Cost-effective model selection with automatic fallback
        # Environment variable OPENAI_MODEL controls choice (defaults to cheapest: gpt-3.5-turbo)
        self.preferred_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # Model capabilities mapping for conditional features
        self.model_capabilities = {
            "gpt-4-turbo": {"supports_json_mode": True, "cost_tier": "high"},
            "gpt-4o": {"supports_json_mode": True, "cost_tier": "high"},
            "gpt-4o-mini": {"supports_json_mode": True, "cost_tier": "low"},
            "gpt-3.5-turbo": {"supports_json_mode": False, "cost_tier": "low"}
        }
        
        # Dynamically build fallback chain with preferred model first
        all_models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]
        self.model_fallbacks = self._build_fallback_chain(self.preferred_model, all_models)
        self.current_model = self.preferred_model
        
        # Batch processing configuration - optimized for speed
        self.batch_size = 5  # Process 5 opportunities per API call (true batching)
        self.concurrent_batches = 3  # Number of concurrent API calls
        self.max_retries = 2  # Reduced retries for speed
        self.base_delay = 0.5  # Faster retry delay
        self.timeout = 90  # Increased timeout for batch processing (OpenAI can take 30-60s)
        
        # Secure API key handling
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        
        self.client = OpenAI(api_key=api_key, timeout=self.timeout)
        
        logger.info(f"💰 OpenAI client initialized with cost-effective model chain: {self.model_fallbacks}")
        logger.info(f"🎯 Preferred model: {self.preferred_model} (set OPENAI_MODEL env var to control)")
        logger.info(f"💡 Model capabilities: JSON mode={self._supports_json_mode(self.preferred_model)}, Cost tier={self._get_cost_tier(self.preferred_model)}")

    def _build_fallback_chain(self, preferred_model: str, all_models: list) -> list:
        """Build fallback chain with preferred model first, then by cost efficiency"""
        if preferred_model not in all_models:
            logger.warning(f"⚠️ Preferred model '{preferred_model}' not in supported models, using default chain")
            return all_models
        
        # Remove preferred model from list and put it first
        fallback_chain = [preferred_model]
        remaining_models = [m for m in all_models if m != preferred_model]
        
        # Sort remaining by cost (low cost first for better fallback)
        remaining_models.sort(key=lambda m: self.model_capabilities.get(m, {}).get('cost_tier', 'high') == 'low', reverse=True)
        fallback_chain.extend(remaining_models)
        
        return fallback_chain
    
    def _supports_json_mode(self, model_name: str) -> bool:
        """Check if a model supports structured JSON response format"""
        return self.model_capabilities.get(model_name, {}).get('supports_json_mode', False)
    
    def _get_cost_tier(self, model_name: str) -> str:
        """Get cost tier for a model"""
        return self.model_capabilities.get(model_name, {}).get('cost_tier', 'unknown')

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
            
            # Parse additional scoring details if available  
            meddpicc_score = result.get('meddpicc_score', score)
            bant_score = result.get('bant_score', score) 
            key_strengths = result.get('key_strengths', 'Not specified')
            areas_for_improvement = result.get('areas_for_improvement', 'Not specified')
            confidence_level = result.get('confidence_level', 'Medium')
            
            return {
                'Opportunity_ID': opportunity_row.get('Opportunity_ID', None),  # Preserve ID for linking
                'Opportunity Name': opportunity_row.get('Opportunity Name', 'Unknown'),
                'Account Name': opportunity_row.get('Account Name', 'Unknown'),  
                'Score': score,
                'Reasoning': explanation,
                'MEDDPICC Score': meddpicc_score,
                'BANT Score': bant_score,
                'Key Strengths': key_strengths,
                'Areas for Improvement': areas_for_improvement,
                'Confidence Level': confidence_level,
                'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error scoring opportunity: {str(e)}")
            return {
                'Opportunity_ID': opportunity_row.get('Opportunity_ID', None),  # Preserve ID for linking
                'Opportunity Name': opportunity_row.get('Opportunity Name', 'Unknown'),
                'Account Name': opportunity_row.get('Account Name', 'Unknown'),
                'Score': 0,
                'Reasoning': f"Scoring error: {str(e)}",
                'MEDDPICC Score': 0,
                'BANT Score': 0,
                'Key Strengths': 'Not available due to error',
                'Areas for Improvement': 'Not available due to error', 
                'Confidence Level': 'Low',
                'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def _prepare_opportunity_data(self, row):
        """Prepare opportunity data for scoring, removing any sensitive info"""
        return {
            'opportunity_name': row.get('Opportunity Name', ''),
            'account': row.get('Account Name', ''),
            'amount': self._safe_amount(row.get('Amount', 0)),
            'stage': row.get('Stage Name', ''),
            'product': row.get('Product', ''),
            'notes': row.get('Opportunity Notes', ''),
            'metrics': row.get('Metrics', ''),
            'economic_buyer': row.get('Economic Buyer', ''),
            'decision_criteria': row.get('Decision Criteria', ''),
            'decision_process': row.get('Decision Process', ''),
            'paper_process': row.get('Paper Process', ''),
            'identify_pain': row.get('Identify Pain', ''),
            'champion': row.get('Champion', ''),
            'competition': row.get('Competition', '')
        }

    def _safe_amount(self, value):
        """Safely convert amount to float, handling NaN values"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _extract_json_array(self, content: str) -> List[Dict]:
        """Robust JSON array extraction with code fence stripping and bracket matching"""
        try:
            # Strip common markdown code fences
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            content = content.strip()
            
            # Find the first array in the content using bracket matching
            start_idx = content.find('[')
            if start_idx == -1:
                raise ValueError("No JSON array found in response")
            
            # Use bracket matching to find the complete array
            bracket_count = 0
            end_idx = start_idx
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break
            
            if bracket_count != 0:
                raise ValueError("Unmatched brackets in JSON array")
            
            # Extract and parse the JSON array
            json_str = content[start_idx:end_idx]
            parsed_array = json.loads(json_str)
            
            if not isinstance(parsed_array, list):
                raise ValueError("Extracted JSON is not an array")
            
            return parsed_array
            
        except Exception as e:
            logger.warning(f"JSON array extraction failed: {str(e)}")
            # Try simple json.loads as fallback
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    # Single item wrapped in array
                    return [parsed]
                else:
                    raise ValueError("Parsed JSON is neither array nor object")
            except Exception as fallback_e:
                logger.error(f"Fallback JSON parsing also failed: {str(fallback_e)}")
                raise ValueError(f"All JSON parsing attempts failed: {str(e)}, {str(fallback_e)}")

    def _validate_json_array_items(self, json_array: List[Dict], expected_count: int) -> List[Dict]:
        """Validate and sanitize JSON array items"""
        validated_items = []
        
        for i, item in enumerate(json_array):
            if not isinstance(item, dict):
                logger.warning(f"Item {i} is not a dictionary, skipping")
                continue
                
            # Ensure required fields exist with defaults
            validated_item = {
                'id': item.get('id', i + 1),
                'score': max(0, min(100, int(item.get('score', 0)))),
                'explanation': item.get('explanation', 'No explanation provided')
            }
            
            # Optional fields with defaults
            validated_item.update({
                'meddpicc_score': max(0, min(100, int(item.get('meddpicc_score', validated_item['score'])))),
                'bant_score': max(0, min(100, int(item.get('bant_score', validated_item['score'])))),
                'key_strengths': item.get('key_strengths', 'See explanation'),
                'areas_for_improvement': item.get('areas_for_improvement', 'See explanation'),
                'confidence_level': item.get('confidence_level', 'Medium')
            })
            
            validated_items.append(validated_item)
        
        # Pad with default items if we don't have enough
        while len(validated_items) < expected_count:
            validated_items.append({
                'id': len(validated_items) + 1,
                'score': 0,
                'explanation': 'Missing result from batch response',
                'meddpicc_score': 0,
                'bant_score': 0,
                'key_strengths': 'Not available due to missing result',
                'areas_for_improvement': 'Not available due to missing result',
                'confidence_level': 'Low'
            })
        
        return validated_items[:expected_count]  # Trim to expected count

    def _fallback_batch_scoring(self, opportunities_batch: pd.DataFrame, original_error: str) -> List[Dict]:
        """Graceful fallback scoring system with multiple strategies"""
        batch_size = len(opportunities_batch)
        logger.warning(f"Batch scoring failed, trying fallback strategies for {batch_size} opportunities")
        
        # Strategy 1: Retry with strict JSON-only instruction
        if batch_size <= 5:  # Only try strict retry for small batches
            try:
                logger.info("Fallback strategy 1: Retry with strict JSON-only prompt")
                result = self._retry_batch_strict(opportunities_batch)
                if result:
                    logger.info("Strict retry successful")
                    return result
            except Exception as e:
                logger.warning(f"Strict retry failed: {str(e)}")
        
        # Strategy 2: Split into smaller sub-batches
        if batch_size > 1:
            try:
                logger.info("Fallback strategy 2: Split into smaller sub-batches")
                return self._split_batch_scoring(opportunities_batch)
            except Exception as e:
                logger.warning(f"Sub-batch scoring failed: {str(e)}")
        
        # Strategy 3: Single-item scoring (guaranteed to work)
        logger.info("Fallback strategy 3: Single-item scoring")
        return self._single_item_fallback(opportunities_batch)

    def _retry_batch_strict(self, opportunities_batch: pd.DataFrame) -> Optional[List[Dict]]:
        """Retry batch with strict JSON-only instruction"""
        batch_data = []
        for _, row in opportunities_batch.iterrows():
            opp_data = self._prepare_opportunity_data(row)
            batch_data.append(opp_data)
        
        # Create stricter prompt
        strict_prompt = self._create_batch_prompt(batch_data)
        strict_prompt += "\n\nIMPORTANT: Return ONLY the JSON array, no additional text, explanations, or code fences."
        
        response = self._make_api_call_with_retry(strict_prompt, is_batch=True)
        
        if not response or not response.choices:
            raise ValueError("Invalid response from strict retry")
            
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from strict retry")
        
        # Try parsing with robust extraction
        batch_results = self._extract_json_array(content)
        validated_results = self._validate_json_array_items(batch_results, len(opportunities_batch))
        
        # Format results
        formatted_results = []
        for i, (_, row) in enumerate(opportunities_batch.iterrows()):
            result_data = validated_results[i] if i < len(validated_results) else validated_results[-1]
            formatted_results.append(self._format_batch_result(row, result_data))
        
        return formatted_results

    def _split_batch_scoring(self, opportunities_batch: pd.DataFrame) -> List[Dict]:
        """Split batch into smaller sub-batches"""
        batch_size = len(opportunities_batch)
        sub_batch_size = max(1, batch_size // 2)  # Split in half
        
        all_results = []
        for i in range(0, batch_size, sub_batch_size):
            sub_batch = opportunities_batch.iloc[i:i + sub_batch_size]
            logger.info(f"Processing sub-batch {i//sub_batch_size + 1} ({len(sub_batch)} opportunities)")
            
            try:
                # Try normal batch scoring for sub-batch
                sub_results = self._score_batch_optimized(sub_batch)
                all_results.extend(sub_results)
            except Exception as e:
                logger.warning(f"Sub-batch failed, falling back to single scoring: {str(e)}")
                # Fall back to single scoring for this sub-batch
                single_results = self._single_item_fallback(sub_batch)
                all_results.extend(single_results)
        
        return all_results

    def _single_item_fallback(self, opportunities_batch: pd.DataFrame) -> List[Dict]:
        """Fall back to single-item scoring (guaranteed to work)"""
        results = []
        for _, row in opportunities_batch.iterrows():
            try:
                result = self.score_opportunity(row)
                results.append(result)
            except Exception as e:
                logger.error(f"Even single scoring failed for {row.get('Opportunity Name', 'Unknown')}: {str(e)}")
                # Create error result
                results.append({
                    'Opportunity_ID': row.get('Opportunity_ID', None),
                    'Opportunity Name': row.get('Opportunity Name', 'Unknown'),
                    'Account Name': row.get('Account Name', 'Unknown'),
                    'Score': 0,
                    'Reasoning': f"Single scoring fallback error: {str(e)}",
                    'MEDDPICC Score': 0,
                    'BANT Score': 0,
                    'Key Strengths': 'Not available due to error',
                    'Areas for Improvement': 'Not available due to error',
                    'Confidence Level': 'Low',
                    'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
        return results

    def _format_batch_result(self, row: pd.Series, result_data: Dict) -> Dict:
        """Format a single batch result into the expected format"""
        score = result_data.get('score', 0)
        return {
            'Opportunity_ID': row.get('Opportunity_ID', None),
            'Opportunity Name': row.get('Opportunity Name', 'Unknown'),
            'Account Name': row.get('Account Name', 'Unknown'),
            'Score': score,
            'Reasoning': result_data.get('explanation', 'No explanation provided'),
            'MEDDPICC Score': result_data.get('meddpicc_score', score),
            'BANT Score': result_data.get('bant_score', score),
            'Key Strengths': result_data.get('key_strengths', 'See reasoning'),
            'Areas for Improvement': result_data.get('areas_for_improvement', 'See reasoning'),
            'Confidence Level': result_data.get('confidence_level', 'Medium'),
            'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
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

    def _make_api_call_with_retry(self, prompt: str, is_batch: bool = False):
        """Make OpenAI API call with model fallback and retry logic to save costs"""
        last_error = None
        
        # Try each model in the fallback chain
        for model_name in self.model_fallbacks:
            cost_tier = self._get_cost_tier(model_name)
            supports_json = self._supports_json_mode(model_name)
            logger.info(f"💰 Trying {cost_tier}-cost model: {model_name} (JSON mode: {supports_json})")
            
            for attempt in range(self.max_retries):
                try:
                    # Base parameters for all models
                    call_params = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system", 
                                "content": "You are an expert sales opportunity evaluator with 20+ years of enterprise experience. Use MEDDPICC and BANT frameworks to provide professional, detailed scoring with actionable insights."
                            },
                            {
                                "role": "user",
                                "content": self._adapt_prompt_for_model(prompt, model_name, is_batch)
                            }
                        ],
                        "temperature": 0.3,  # Consistent scoring
                        "max_tokens": 1000 if is_batch else 500  # More tokens for batch responses
                    }
                    
                    # Conditional JSON mode support - only for compatible models and single scoring
                    if not is_batch and self._supports_json_mode(model_name):
                        call_params["response_format"] = {"type": "json_object"}
                        logger.debug(f"🔧 Using JSON mode for {model_name}")
                    else:
                        logger.debug(f"🔧 Using text mode for {model_name} (batch={is_batch}, supports_json={supports_json})")
                    
                    response = self.client.chat.completions.create(**call_params)
                    
                    # Success! Update current model for future calls
                    self.current_model = model_name
                    logger.info(f"✅ Response generated successfully with {model_name} ({cost_tier} cost)")
                    return response
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()
                    
                    # Enhanced error detection for model incompatibility
                    model_error_terms = ['model', 'unsupported', 'not found', 'quota', 'insufficient', 'response_format', 'json_object']
                    if any(term in error_str for term in model_error_terms):
                        logger.warning(f"⚠️ Model {model_name} failed (likely incompatible): {e}")
                        break  # Try next model immediately
                    
                    # Otherwise, retry with same model
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"API call attempt {attempt + 1} for {model_name} failed: {e}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        logger.warning(f"All retry attempts failed for {model_name}: {e}")
                        break  # Try next model
        
        # All models failed
        raise RuntimeError(f"💸 All models in fallback chain failed! Last error: {last_error}")
    
    def _adapt_prompt_for_model(self, prompt: str, model_name: str, is_batch: bool) -> str:
        """Adapt prompt based on model capabilities for better reliability"""
        if not self._supports_json_mode(model_name) and not is_batch:
            # For models without JSON mode, add stronger formatting instructions
            prompt += "\n\nIMPORTANT: Respond with valid JSON only. Start with { and end with }. No markdown, no explanations outside the JSON."
        
        return prompt

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
                    opp_name = row.get('Opportunity Name', 'Unknown')
                    logger.info(f"Scoring opportunity: {opp_name}")
                    result = self.score_opportunity(row)
                    logger.info(f"Successfully scored: {opp_name} (Score: {result.get('Score', 'N/A')})")
                    batch_results.append(result)
                    
                    if progress_callback:
                        progress_callback(batch_start + len(batch_results), total_opportunities)
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to score opportunity {row.get('Opportunity Name', 'Unknown')}: {str(e)}")
                    # Add failed result
                    batch_results.append({
                        'Opportunity_ID': row.get('Opportunity_ID', None),  # Preserve ID for linking
                        'Opportunity Name': row.get('Opportunity Name', 'Unknown'),
                        'Account Name': row.get('Account Name', 'Unknown'),
                        'Score': 0,
                        'Reasoning': f"Scoring failed: {str(e)}",
                        'MEDDPICC Score': 0,
                        'BANT Score': 0,
                        'Key Strengths': 'Not available due to error',
                        'Areas for Improvement': 'Not available due to error',
                        'Confidence Level': 'Low',
                        'Scoring_Timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    if progress_callback:
                        progress_callback(batch_start + len(batch_results), total_opportunities)
            
            results.extend(batch_results)
            
            # Add small delay between batches to avoid rate limiting
            if batch_end < total_opportunities:
                time.sleep(0.5)
        
        # Count actual failures by checking results (since score_opportunity swallows errors)
        actual_failures = sum(1 for result in results if result['Score'] == 0 and 'error' in result['Reasoning'].lower())
        success_count = total_opportunities - actual_failures
        success_rate = (success_count / total_opportunities) * 100
        
        logger.info(f"Batch scoring completed. Success rate: {success_rate:.1f}% ({success_count}/{total_opportunities})")
        
        # Save results to database
        try:
            db_manager = get_db_manager()
            scoring_batch_id = f"scoring_{time.strftime('%Y%m%d_%H%M%S')}"
            saved_count = db_manager.save_opportunity_scores(results, scoring_batch_id)
            logger.info(f"Saved {saved_count} opportunity scores to database with batch_id: {scoring_batch_id}")
        except Exception as e:
            logger.error(f"Failed to save opportunity scores to database: {str(e)}")
            # Continue without failing - still return results for backward compatibility
        
        return results, actual_failures

    def _create_batch_prompt(self, opportunities_data: List[Dict]) -> str:
        """Create optimized batch prompt for multiple opportunities"""
        
        def safe_str(value, max_len=50):
            """Safely convert value to string and truncate, handling NaN values"""
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return 'N/A'
            return str(value)[:max_len]
        
        def safe_amount(value):
            """Safely convert amount to float, handling NaN values"""
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return 0.0
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        prompt = """Score these cybersecurity opportunities (0-100) using MEDDPICC/BANT criteria:

OPPORTUNITIES:"""
        
        for i, opp in enumerate(opportunities_data, 1):
            prompt += f"""
{i}. {safe_str(opp['opportunity_name'], 40)} | {safe_str(opp['account'], 30)} | ${safe_amount(opp['amount']):,.0f} | {safe_str(opp['stage'], 25)}
   MEDDPICC: M={safe_str(opp['metrics'], 40)} | E={safe_str(opp['economic_buyer'], 25)} | P={safe_str(opp['identify_pain'], 40)}"""

        prompt += """

Return JSON array:
[{"id": 1, "score": 85, "explanation": "Strong MEDDPICC..."}, {"id": 2, "score": 65, "explanation": "..."}, ...]

Score based on: MEDDPICC completeness (40%), business value (25%), buyer engagement (20%), process clarity (15%)."""
        
        return prompt

    def _score_batch_optimized(self, opportunities_batch: pd.DataFrame) -> List[Dict]:
        """Score a batch of opportunities in a single API call"""
        try:
            # Prepare batch data with compact format
            batch_data = []
            for _, row in opportunities_batch.iterrows():
                opp_data = self._prepare_opportunity_data(row)
                batch_data.append(opp_data)
            
            # Create compact batch prompt
            prompt = self._create_batch_prompt(batch_data)
            
            # Make single API call for entire batch (allow array response)
            response = self._make_api_call_with_retry(prompt, is_batch=True)
            
            if not response or not response.choices:
                raise ValueError("Invalid response from OpenAI API")
                
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI API")
                
            # Parse batch results with robust JSON extraction
            batch_results = self._extract_json_array(content)
            validated_results = self._validate_json_array_items(batch_results, len(opportunities_batch))
            
            # Format results
            formatted_results = []
            for i, (_, row) in enumerate(opportunities_batch.iterrows()):
                result_data = validated_results[i] if i < len(validated_results) else validated_results[-1]
                formatted_results.append(self._format_batch_result(row, result_data))
            
            return formatted_results
            
        except Exception as e:
            logger.warning(f"Primary batch scoring failed: {str(e)}")
            # Use graceful fallback system
            return self._fallback_batch_scoring(opportunities_batch, str(e))

    def score_opportunities_optimized(self, opportunities_df, batch_size=None, progress_callback=None):
        """Optimized scoring with true batching and concurrent processing"""
        total_opportunities = len(opportunities_df)
        
        if total_opportunities == 0:
            return [], 0
        
        # Use provided batch_size or fall back to default
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        
        logger.info(f"🚀 Starting OPTIMIZED batch scoring for {total_opportunities} opportunities")
        logger.info(f"Configuration: {effective_batch_size} opps/batch, {self.concurrent_batches} concurrent batches")
        
        start_time = time.time()
        
        # Split into batches for true batch processing
        batches = []
        for i in range(0, total_opportunities, effective_batch_size):
            batch = opportunities_df.iloc[i:i + effective_batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches for processing")
        
        all_results = []
        failed_count = 0
        
        # Process batches with controlled concurrency
        with ThreadPoolExecutor(max_workers=self.concurrent_batches) as executor:
            # Submit batch jobs
            future_to_batch = {}
            for batch_idx, batch in enumerate(batches):
                future = executor.submit(self._score_batch_optimized, batch)
                future_to_batch[future] = batch_idx
            
            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                completed_batches += 1
                
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Count failures in this batch
                    batch_failures = sum(1 for r in batch_results if r['Score'] == 0)
                    failed_count += batch_failures
                    
                    logger.info(f"✅ Batch {completed_batches}/{len(batches)} completed ({len(batch_results)} opportunities)")
                    
                    if progress_callback:
                        progress_callback(len(all_results), total_opportunities)
                        
                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx + 1} failed: {str(e)}")
                    failed_count += len(batches[batch_idx])
        
        end_time = time.time()
        processing_time = end_time - start_time
        success_count = total_opportunities - failed_count
        success_rate = (success_count / total_opportunities) * 100
        
        logger.info(f"🎯 OPTIMIZED scoring completed in {processing_time:.1f} seconds!")
        logger.info(f"Performance: {total_opportunities/processing_time:.1f} opportunities/second")
        logger.info(f"Success rate: {success_rate:.1f}% ({success_count}/{total_opportunities})")
        
        # Save results to database
        try:
            db_manager = get_db_manager()
            scoring_batch_id = f"scoring_optimized_{time.strftime('%Y%m%d_%H%M%S')}"
            saved_count = db_manager.save_opportunity_scores(all_results, scoring_batch_id)
            logger.info(f"💾 Saved {saved_count} opportunity scores to database")
        except Exception as e:
            logger.error(f"Failed to save to database: {str(e)}")
        
        return all_results, failed_count
