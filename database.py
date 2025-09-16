"""
Database module for PostgreSQL operations with SQLAlchemy ORM
Handles all database connections, schema creation, and data operations
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Dict, Optional, Any
import json

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration
Base = declarative_base()

class SyntheticOpportunity(Base):
    """SQLAlchemy model for synthetic opportunities data"""
    __tablename__ = 'synthetic_opportunities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_name = Column(String(255))
    account_name = Column(String(255))
    stage_name = Column(String(100))
    amount = Column(Float)
    close_date = Column(String(50))  # Keep as string to match CSV format
    probability = Column(Float)
    owner_name = Column(String(255))
    lead_source = Column(String(100))
    opportunity_type = Column(String(100))
    
    # MEDDPICC fields
    metrics = Column(Text)
    economic_buyer = Column(String(255))
    decision_criteria = Column(Text)
    decision_process = Column(Text)
    paper_process = Column(Text)
    identify_pain = Column(Text)
    champion = Column(String(255))
    competition = Column(String(255))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    batch_id = Column(String(100))

class OpportunityScore(Base):
    """SQLAlchemy model for AI opportunity scores"""
    __tablename__ = 'opportunity_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    opportunity_name = Column(String(255))
    account_name = Column(String(255))
    score = Column(Integer)
    reasoning = Column(Text)
    meddpicc_score = Column(Integer)
    bant_score = Column(Integer)
    key_strengths = Column(Text)
    areas_for_improvement = Column(Text)
    confidence_level = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    scoring_batch_id = Column(String(100))

class DatabaseManager:
    """Database manager for PostgreSQL operations"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        self.engine = create_engine(self.database_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("Database manager initialized successfully")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
    
    def get_session(self) -> Session:
        """Get a database session"""
        return self.Session()
    
    def save_synthetic_opportunities(self, df: pd.DataFrame, batch_id: str = None) -> int:
        """Save synthetic opportunities DataFrame to database"""
        session = self.get_session()
        try:
            if batch_id is None:
                batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            saved_count = 0
            for _, row in df.iterrows():
                opportunity = SyntheticOpportunity(
                    opportunity_name=row.get('Opportunity Name') or '',
                    account_name=row.get('Account Name') or '',
                    stage_name=row.get('Stage Name') or '',
                    amount=float(row.get('Amount', 0)) if row.get('Amount') is not None else 0,
                    close_date=str(row.get('Close Date') or ''),
                    probability=float(row.get('Probability', 0)) if row.get('Probability') is not None else 0,
                    owner_name=row.get('Owner Name') or '',
                    lead_source=row.get('Lead Source') or '',
                    opportunity_type=row.get('Type') or '',
                    metrics=row.get('Metrics'),
                    economic_buyer=row.get('Economic Buyer'),
                    decision_criteria=row.get('Decision Criteria'),
                    decision_process=row.get('Decision Process'),
                    paper_process=row.get('Paper Process'),
                    identify_pain=row.get('Identify Pain'),
                    champion=row.get('Champion'),
                    competition=row.get('Competition'),
                    batch_id=batch_id
                )
                session.add(opportunity)
                saved_count += 1
            
            session.commit()
            logger.info(f"Saved {saved_count} synthetic opportunities to database")
            return saved_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving synthetic opportunities: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_synthetic_opportunities(self, batch_id: str = None, limit: int = None) -> pd.DataFrame:
        """Load synthetic opportunities from database"""
        session = self.get_session()
        try:
            query = session.query(SyntheticOpportunity)
            
            if batch_id:
                query = query.filter(SyntheticOpportunity.batch_id == batch_id)
            
            if limit:
                query = query.limit(limit)
            
            opportunities = query.all()
            
            # Convert to DataFrame
            data = []
            for opp in opportunities:
                data.append({
                    'Opportunity Name': opp.opportunity_name,
                    'Account Name': opp.account_name,
                    'Stage Name': opp.stage_name,
                    'Amount': opp.amount,
                    'Close Date': opp.close_date,
                    'Probability': opp.probability,
                    'Owner Name': opp.owner_name,
                    'Lead Source': opp.lead_source,
                    'Type': opp.opportunity_type,
                    'Metrics': opp.metrics,
                    'Economic Buyer': opp.economic_buyer,
                    'Decision Criteria': opp.decision_criteria,
                    'Decision Process': opp.decision_process,
                    'Paper Process': opp.paper_process,
                    'Identify Pain': opp.identify_pain,
                    'Champion': opp.champion,
                    'Competition': opp.competition,
                    'Created At': opp.created_at,
                    'Batch ID': opp.batch_id
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} synthetic opportunities from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading synthetic opportunities: {str(e)}")
            raise
        finally:
            session.close()
    
    def save_opportunity_scores(self, scores_data: List[Dict], scoring_batch_id: str = None) -> int:
        """Save opportunity scores to database"""
        session = self.get_session()
        try:
            if scoring_batch_id is None:
                scoring_batch_id = f"scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            saved_count = 0
            for score_data in scores_data:
                score = OpportunityScore(
                    opportunity_name=score_data.get('Opportunity Name') or '',
                    account_name=score_data.get('Account Name') or '',
                    score=int(score_data.get('Score', 0)) if score_data.get('Score') is not None else 0,
                    reasoning=score_data.get('Reasoning'),
                    meddpicc_score=int(score_data.get('MEDDPICC Score', 0)) if score_data.get('MEDDPICC Score') is not None else 0,
                    bant_score=int(score_data.get('BANT Score', 0)) if score_data.get('BANT Score') is not None else 0,
                    key_strengths=score_data.get('Key Strengths'),
                    areas_for_improvement=score_data.get('Areas for Improvement'),
                    confidence_level=score_data.get('Confidence Level'),
                    scoring_batch_id=scoring_batch_id
                )
                session.add(score)
                saved_count += 1
            
            session.commit()
            logger.info(f"Saved {saved_count} opportunity scores to database")
            return saved_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving opportunity scores: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_opportunity_scores(self, scoring_batch_id: str = None, limit: int = None) -> pd.DataFrame:
        """Load opportunity scores from database"""
        session = self.get_session()
        try:
            query = session.query(OpportunityScore)
            
            if scoring_batch_id:
                query = query.filter(OpportunityScore.scoring_batch_id == scoring_batch_id)
            
            if limit:
                query = query.limit(limit)
                
            # Order by most recent first
            query = query.order_by(OpportunityScore.created_at.desc())
            
            scores = query.all()
            
            # Convert to DataFrame
            data = []
            for score in scores:
                data.append({
                    'Opportunity Name': score.opportunity_name,
                    'Account Name': score.account_name,
                    'Score': score.score,
                    'Reasoning': score.reasoning,
                    'MEDDPICC Score': score.meddpicc_score,
                    'BANT Score': score.bant_score,
                    'Key Strengths': score.key_strengths,
                    'Areas for Improvement': score.areas_for_improvement,
                    'Confidence Level': score.confidence_level,
                    'Created At': score.created_at,
                    'Scoring Batch ID': score.scoring_batch_id
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} opportunity scores from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading opportunity scores: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_available_batches(self) -> Dict[str, List[str]]:
        """Get available batch IDs for opportunities and scores"""
        session = self.get_session()
        try:
            # Get opportunity batches
            opp_batches = session.query(SyntheticOpportunity.batch_id).distinct().all()
            opp_batch_ids = [batch[0] for batch in opp_batches if batch[0]]
            
            # Get scoring batches
            score_batches = session.query(OpportunityScore.scoring_batch_id).distinct().all()
            score_batch_ids = [batch[0] for batch in score_batches if batch[0]]
            
            return {
                'opportunity_batches': opp_batch_ids,
                'scoring_batches': score_batch_ids
            }
            
        except Exception as e:
            logger.error(f"Error getting available batches: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        session = self.get_session()
        try:
            # Count opportunities
            opp_count = session.query(SyntheticOpportunity).count()
            
            # Count scores
            score_count = session.query(OpportunityScore).count()
            
            # Get latest batch info
            latest_opp_batch = session.query(SyntheticOpportunity.batch_id, 
                                           SyntheticOpportunity.created_at).order_by(
                                           SyntheticOpportunity.created_at.desc()).first()
            
            latest_score_batch = session.query(OpportunityScore.scoring_batch_id, 
                                             OpportunityScore.created_at).order_by(
                                             OpportunityScore.created_at.desc()).first()
            
            return {
                'total_opportunities': opp_count,
                'total_scores': score_count,
                'latest_opportunity_batch': latest_opp_batch[0] if latest_opp_batch else None,
                'latest_opportunity_date': latest_opp_batch[1] if latest_opp_batch else None,
                'latest_scoring_batch': latest_score_batch[0] if latest_score_batch else None,
                'latest_scoring_date': latest_score_batch[1] if latest_score_batch else None
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise
        finally:
            session.close()

# Global database manager instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get or create database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
        db_manager.create_tables()
    return db_manager

def init_database():
    """Initialize database and create tables"""
    try:
        db_manager = get_db_manager()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False