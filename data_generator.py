import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
import logging
from database import get_db_manager

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Secure synthetic Salesforce opportunity data generator"""
    
    def __init__(self):
        self.fake = Faker()
        Faker.seed(42)  # For reproducible results
        random.seed(42)
        np.random.seed(42)
        
        # Public company names (Fortune 500 subset)
        self.public_companies = [
            "Apple Inc.", "Microsoft Corporation", "Amazon.com Inc.", "Alphabet Inc.",
            "Tesla Inc.", "Meta Platforms Inc.", "NVIDIA Corporation", "Berkshire Hathaway Inc.",
            "Johnson & Johnson", "JPMorgan Chase & Co.", "Procter & Gamble Co.",
            "UnitedHealth Group Inc.", "Home Depot Inc.", "Mastercard Inc.", "Bank of America Corp.",
            "Pfizer Inc.", "Chevron Corporation", "Coca-Cola Company", "PepsiCo Inc.",
            "Walt Disney Company", "Comcast Corporation", "Cisco Systems Inc.", "Intel Corporation",
            "Verizon Communications Inc.", "AT&T Inc.", "Oracle Corporation", "Salesforce Inc.",
            "Netflix Inc.", "Adobe Inc.", "PayPal Holdings Inc.", "Broadcom Inc.",
            "Texas Instruments Inc.", "Qualcomm Inc.", "Advanced Micro Devices Inc.", "Intuit Inc.",
            "American Express Company", "Goldman Sachs Group Inc.", "Morgan Stanley", "Wells Fargo & Co.",
            "Citigroup Inc.", "IBM Corporation", "General Electric Company", "Boeing Company",
            "Caterpillar Inc.", "3M Company", "McDonald's Corporation", "Starbucks Corporation",
            "Nike Inc.", "Ford Motor Company", "General Motors Company"
        ]
        
        # CrowdStrike products/services
        self.crowdstrike_products = [
            "Falcon Platform", "Falcon Prevent", "Falcon Insight", "Falcon Complete",
            "Falcon X", "Falcon Intelligence", "Falcon OverWatch", "Falcon Discover",
            "Falcon Spotlight", "CrowdStrike Store", "Professional Services"
        ]
        
        # Sales stages
        self.stages = [
            "Stage 1 - Prospecting", "Stage 1 - Qualification", "Stage 1 - Needs Analysis",
            "Stage 1 - Proposal", "Stage 1 - Negotiation"
        ]
        
        # Sales owners (synthetic)
        self.sales_owners = [
            "Sarah Chen", "Michael Rodriguez", "Jennifer Park", "David Thompson",
            "Lisa Wang", "Robert Johnson", "Maria Garcia", "James Wilson",
            "Amanda Davis", "Christopher Lee", "Rachel Martinez", "Kevin Brown"
        ]
        
        # MEDDPICC criteria options
        self.decision_criteria = [
            "Security effectiveness", "Total cost of ownership", "Ease of deployment",
            "Integration capabilities", "Compliance requirements", "Vendor stability",
            "Support quality", "Scalability", "Performance impact"
        ]
        
        self.competitors = [
            "SentinelOne", "Carbon Black", "Cylance", "Symantec", "McAfee",
            "Trend Micro", "Palo Alto Networks", "Fortinet", "Check Point"
        ]

    def generate_opportunities(self, count=50, batch_size=None):
        """Generate synthetic opportunities with MEDDPICC fields"""
        logger.info(f"Generating {count} synthetic opportunities")
        
        # Set batch size for large datasets
        if batch_size is None:
            batch_size = min(100, count)  # Process in batches of 100 for large datasets
        
        opportunities = []
        
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(count-1)//batch_size + 1} ({batch_start+1}-{batch_end} of {count})")
            
            for i in range(batch_start, batch_end):
                # Basic opportunity data
                company = random.choice(self.public_companies)
                product = random.choice(self.crowdstrike_products)
                
                # Generate unique opportunity name
                opp_name = f"{company} - {product} Implementation"
                
                # Random dates
                created_date = self.fake.date_between(start_date='-6M', end_date='today')
                close_date = created_date + timedelta(days=random.randint(30, 180))
                
                # Amount (realistic enterprise security deals)
                amount = random.choice([
                    random.randint(25000, 100000),    # SMB
                    random.randint(100000, 500000),   # Mid-market
                    random.randint(500000, 2000000)   # Enterprise
                ])
                
                # Opportunity notes
                notes = self._generate_opportunity_notes(company, product)
                
                opportunity = {
                    'Opportunity Name': opp_name,
                    'Account Name': company,
                    'Amount': amount,
                    'Close Date': close_date.strftime('%Y-%m-%d'),
                    'Stage Name': random.choice(self.stages),
                    'Owner Name': random.choice(self.sales_owners),
                    'Lead Source': random.choice(['Website', 'Referral', 'Cold Call', 'Trade Show', 'Partner']),
                    'Type': 'New Business',
                    'Probability': random.randint(10, 90),
                    'Product': product,
                    'Opportunity Notes': notes
                }
                
                # Add MEDDPICC fields for ~50% of opportunities
                if random.random() < 0.5:
                    opportunity.update(self._generate_meddpicc_fields())
                else:
                    # Partially populate to reflect reality
                    opportunity.update(self._generate_partial_meddpicc())
                
                opportunities.append(opportunity)
        
        df = pd.DataFrame(opportunities)
        logger.info(f"Successfully generated {len(df)} synthetic opportunities")
        
        # Save to database
        try:
            db_manager = get_db_manager()
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_count = db_manager.save_synthetic_opportunities(df, batch_id)
            logger.info(f"Saved {saved_count} opportunities to database with batch_id: {batch_id}")
        except Exception as e:
            logger.error(f"Failed to save opportunities to database: {str(e)}")
            # Continue without failing - still return DataFrame for backward compatibility
        
        return df

    def _generate_opportunity_notes(self, company, product):
        """Generate realistic opportunity notes (Why? Why now? Why this product?)"""
        why_reasons = [
            "Increasing cyber threats and compliance requirements",
            "Recent security incidents highlighting gaps in current protection",
            "Digital transformation initiative requiring enhanced security",
            "Regulatory compliance mandates (SOX, GDPR, HIPAA)",
            "Merger/acquisition requiring security consolidation"
        ]
        
        why_now_reasons = [
            "Current security contract expires in Q2",
            "Recent ransomware attack on industry peer",
            "New CISO mandate for next-gen security solutions",
            "Budget approved for security infrastructure upgrade",
            "Compliance audit findings require immediate action"
        ]
        
        why_product_reasons = [
            f"{product} offers superior threat detection capabilities",
            "Cloud-native architecture aligns with IT strategy",
            "Strong ROI demonstrated in competitor analysis",
            "Excellent references from similar industry companies",
            "Comprehensive platform reduces vendor complexity"
        ]
        
        why = random.choice(why_reasons)
        why_now = random.choice(why_now_reasons)
        why_product = random.choice(why_product_reasons)
        
        return f"WHY: {why}. WHY NOW: {why_now}. WHY THIS PRODUCT: {why_product}."

    def _generate_meddpicc_fields(self):
        """Generate complete MEDDPICC fields"""
        return {
            'Metrics': f"Reduce security incidents by {random.randint(30, 70)}%, improve MTTR by {random.randint(40, 80)}%",
            'Economic Buyer': f"{random.choice(['CISO', 'CTO', 'CEO', 'CFO'])} - {self.fake.name()}",
            'Decision Criteria': ', '.join(random.sample(self.decision_criteria, 3)),
            'Decision Process': f"{random.randint(60, 120)} day evaluation, {random.randint(3, 7)} stakeholders",
            'Paper Process': f"{random.choice(['Legal review', 'Procurement approval', 'Security assessment'])} required",
            'Identify Pain': f"Current solution has {random.choice(['high false positives', 'slow response times', 'limited visibility', 'complex management'])}",
            'Champion': f"{self.fake.name()} - {random.choice(['Security Analyst', 'IT Director', 'SOC Manager'])}",
            'Competition': ', '.join(random.sample(self.competitors, 2))
        }

    def _generate_partial_meddpicc(self):
        """Generate partially populated MEDDPICC fields"""
        from typing import Optional
        fields: dict[str, Optional[str]] = {
            'Metrics': None,
            'Economic Buyer': None,
            'Decision Criteria': None,
            'Decision Process': None,
            'Paper Process': None,
            'Identify Pain': None,
            'Champion': None,
            'Competition': None
        }
        
        # Randomly populate 2-4 fields
        populated_count = random.randint(2, 4)
        fields_to_populate = random.sample(list(fields.keys()), populated_count)
        
        full_meddpicc = self._generate_meddpicc_fields()
        for field in fields_to_populate:
            fields[field] = full_meddpicc[field]
        
        return fields
