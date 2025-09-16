"""
Comprehensive Audit Trail and Monitoring System
Provides tamper-proof audit logging, real-time monitoring, and compliance reporting
for enterprise cybersecurity evaluation.
"""

import json
import time
import hashlib
import hmac
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import sqlite3
import logging
from pathlib import Path

# Configure logger for audit system
audit_logger = logging.getLogger('audit_trail')

class EventType(Enum):
    """Audit event types for classification"""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    DATA_ACCESS = "data_access"
    API_CALL = "api_call"
    ERROR_EVENT = "error_event"
    COMPLIANCE_EVENT = "compliance_event"

class SeverityLevel(Enum):
    """Event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Structured audit event with full context"""
    event_id: str
    timestamp: str
    event_type: EventType
    action: str
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    resource: Optional[str]
    details: Dict[str, Any]
    severity: SeverityLevel
    compliance_tags: List[str]
    data_classification: str
    outcome: str  # SUCCESS, FAILURE, PARTIAL
    duration_ms: Optional[int]
    checksum: str

class AuditTrailManager:
    """Manages tamper-proof audit trails with cryptographic integrity"""
    
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.secret_key = self._get_or_create_secret()
        self._init_database()
        self._setup_retention_policy()
        
    def _get_or_create_secret(self) -> str:
        """Get or create persistent secret key for HMAC integrity"""
        secret = os.environ.get('AUDIT_SECRET_KEY')
        if not secret:
            # For production security, require persistent key
            secret_file = Path('.audit_secret')
            if secret_file.exists():
                try:
                    with open(secret_file, 'r') as f:
                        secret = f.read().strip()
                    audit_logger.info("Loaded persistent audit secret key")
                except Exception as e:
                    audit_logger.error(f"Failed to load persistent secret: {e}")
                    secret = None
            
            if not secret:
                # Generate persistent secret for this deployment
                import secrets
                secret = secrets.token_hex(32)
                try:
                    with open(secret_file, 'w') as f:
                        f.write(secret)
                    secret_file.chmod(0o600)  # Owner read/write only
                    audit_logger.warning("Generated new persistent audit secret key. Store AUDIT_SECRET_KEY in environment for production.")
                except Exception as e:
                    audit_logger.error(f"Failed to persist secret key: {e}. Using session-temporary key.")
        
        return secret
    
    def _init_database(self):
        """Initialize SQLite database for audit trail storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create audit events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                action TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                resource TEXT,
                details TEXT NOT NULL,
                severity TEXT NOT NULL,
                compliance_tags TEXT,
                data_classification TEXT NOT NULL,
                outcome TEXT NOT NULL,
                duration_ms INTEGER,
                checksum TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for efficient queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)')
        
        # Create audit integrity table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_integrity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_start_time TEXT NOT NULL,
                batch_end_time TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                batch_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        audit_logger.info("Audit database initialized successfully")
    
    def _setup_retention_policy(self):
        """Setup audit log retention policy (default: 7 years for compliance)"""
        self.retention_days = int(os.environ.get('AUDIT_RETENTION_DAYS', 2555))  # ~7 years
        
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate HMAC-SHA256 checksum for tamper detection"""
        # Create deterministic string from event data (excluding checksum)
        event_data = {k: v for k, v in asdict(event).items() if k != 'checksum'}
        
        # Convert enums to strings for JSON serialization
        if 'event_type' in event_data and hasattr(event_data['event_type'], 'value'):
            event_data['event_type'] = event_data['event_type'].value
        if 'severity' in event_data and hasattr(event_data['severity'], 'value'):
            event_data['severity'] = event_data['severity'].value
            
        event_string = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
        
        # Calculate HMAC
        return hmac.new(
            self.secret_key.encode('utf-8'),
            event_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def log_event(self, 
                  event_type: EventType,
                  action: str,
                  user_id: str = "system",
                  session_id: str = "default",
                  ip_address: str = "127.0.0.1",
                  user_agent: str = "streamlit-app",
                  resource: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  severity: SeverityLevel = SeverityLevel.LOW,
                  compliance_tags: Optional[List[str]] = None,
                  data_classification: str = "internal",
                  outcome: str = "SUCCESS",
                  duration_ms: Optional[int] = None) -> str:
        """Log audit event with tamper-proof checksum"""
        
        # Generate unique event ID
        event_id = f"evt_{int(time.time() * 1000000)}_{hashlib.md5(action.encode()).hexdigest()[:8]}"
        
        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            action=action,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            details=details or {},
            severity=severity,
            compliance_tags=compliance_tags or [],
            data_classification=data_classification,
            outcome=outcome,
            duration_ms=duration_ms,
            checksum=""  # Will be calculated below
        )
        
        # Calculate and set checksum
        event.checksum = self._calculate_checksum(event)
        
        # Store in database
        self._store_event(event)
        
        # Log to application logger based on severity
        log_message = f"AUDIT: {action} by {user_id} - {outcome}"
        if severity == SeverityLevel.CRITICAL:
            audit_logger.critical(log_message)
        elif severity == SeverityLevel.HIGH:
            audit_logger.error(log_message)
        elif severity == SeverityLevel.MEDIUM:
            audit_logger.warning(log_message)
        else:
            audit_logger.info(log_message)
        
        return event_id
    
    def _store_event(self, event: AuditEvent):
        """Store audit event in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_events 
            (event_id, timestamp, event_type, action, user_id, session_id, 
             ip_address, user_agent, resource, details, severity, compliance_tags,
             data_classification, outcome, duration_ms, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.timestamp,
            event.event_type.value,
            event.action,
            event.user_id,
            event.session_id,
            event.ip_address,
            event.user_agent,
            event.resource,
            json.dumps(event.details),
            event.severity.value,
            json.dumps(event.compliance_tags),
            event.data_classification,
            event.outcome,
            event.duration_ms,
            event.checksum
        ))
        
        conn.commit()
        conn.close()
    
    def verify_integrity(self, hours_back: int = 24) -> Dict[str, Any]:
        """Verify audit trail integrity for the specified time period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get events from the specified time period
        start_time = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
        cursor.execute('''
            SELECT * FROM audit_events 
            WHERE timestamp >= ? 
            ORDER BY timestamp
        ''', (start_time,))
        
        events = cursor.fetchall()
        conn.close()
        
        # Verify checksums
        total_events = len(events)
        tampered_events = []
        verified_events = 0
        
        for row in events:
            # Reconstruct event from database
            event_dict = {
                'event_id': row[0],
                'timestamp': row[1],
                'event_type': EventType(row[2]),
                'action': row[3],
                'user_id': row[4],
                'session_id': row[5],
                'ip_address': row[6],
                'user_agent': row[7],
                'resource': row[8],
                'details': json.loads(row[9]),
                'severity': SeverityLevel(row[10]),
                'compliance_tags': json.loads(row[11]),
                'data_classification': row[12],
                'outcome': row[13],
                'duration_ms': row[14],
                'checksum': ""  # Will calculate fresh
            }
            
            event = AuditEvent(**event_dict)
            stored_checksum = row[15]  # Original checksum from DB
            
            # Calculate fresh checksum
            calculated_checksum = self._calculate_checksum(event)
            
            if calculated_checksum == stored_checksum:
                verified_events += 1
            else:
                tampered_events.append({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'action': event.action,
                    'stored_checksum': stored_checksum,
                    'calculated_checksum': calculated_checksum
                })
        
        integrity_result = {
            'verification_timestamp': datetime.utcnow().isoformat(),
            'hours_checked': hours_back,
            'total_events': total_events,
            'verified_events': verified_events,
            'tampered_events': len(tampered_events),
            'integrity_score': (verified_events / total_events * 100) if total_events > 0 else 100,
            'tampered_event_details': tampered_events
        }
        
        # Log integrity check result
        if tampered_events:
            self.log_event(
                EventType.SECURITY_EVENT,
                f"INTEGRITY_VIOLATION_DETECTED",
                severity=SeverityLevel.CRITICAL,
                details={'integrity_result': integrity_result},
                compliance_tags=['SOX', 'GDPR', 'HIPAA']
            )
        else:
            self.log_event(
                EventType.SYSTEM_EVENT,
                f"INTEGRITY_CHECK_PASSED",
                severity=SeverityLevel.LOW,
                details={'events_verified': verified_events}
            )
        
        return integrity_result
    
    def get_events(self, 
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   event_types: Optional[List[EventType]] = None,
                   user_id: Optional[str] = None,
                   severity: Optional[SeverityLevel] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Query audit events with filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        if event_types:
            placeholders = ','.join(['?' for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])
            
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
            
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to dictionaries
        events = []
        column_names = [
            'event_id', 'timestamp', 'event_type', 'action', 'user_id', 
            'session_id', 'ip_address', 'user_agent', 'resource', 'details',
            'severity', 'compliance_tags', 'data_classification', 'outcome',
            'duration_ms', 'checksum', 'created_at'
        ]
        
        for row in rows:
            event_dict = dict(zip(column_names, row))
            # Parse JSON fields
            event_dict['details'] = json.loads(event_dict['details'])
            event_dict['compliance_tags'] = json.loads(event_dict['compliance_tags'])
            events.append(event_dict)
        
        return events
    
    def generate_compliance_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        start_time = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        
        # Get all events for the period
        events = self.get_events(start_time=start_time, limit=10000)
        
        # Analysis
        total_events = len(events)
        event_types = {}
        severity_breakdown = {}
        user_activity = {}
        daily_activity = {}
        security_events = 0
        failed_events = 0
        
        for event in events:
            # Event type distribution
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Severity breakdown
            severity = event['severity']
            severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1
            
            # User activity
            user_id = event['user_id']
            user_activity[user_id] = user_activity.get(user_id, 0) + 1
            
            # Daily activity
            event_date = event['timestamp'][:10]  # Extract date
            daily_activity[event_date] = daily_activity.get(event_date, 0) + 1
            
            # Security and failure metrics
            if event['event_type'] == 'security_event':
                security_events += 1
            if event['outcome'] != 'SUCCESS':
                failed_events += 1
        
        # Verify integrity
        integrity_result = self.verify_integrity(hours_back=days_back * 24)
        
        # Generate report
        report = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'period_days': days_back,
            'total_events': total_events,
            'event_type_distribution': event_types,
            'severity_breakdown': severity_breakdown,
            'top_users': dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            'daily_activity': daily_activity,
            'security_events_count': security_events,
            'failed_events_count': failed_events,
            'success_rate': ((total_events - failed_events) / total_events * 100) if total_events > 0 else 100,
            'integrity_verification': integrity_result,
            'compliance_status': {
                'SOX': 'COMPLIANT' if integrity_result['integrity_score'] == 100 else 'ATTENTION_REQUIRED',
                'GDPR': 'COMPLIANT' if security_events == 0 else 'REVIEW_REQUIRED',
                'HIPAA': 'COMPLIANT' if failed_events / total_events < 0.01 else 'ATTENTION_REQUIRED'
            }
        }
        
        # Log report generation
        self.log_event(
            EventType.COMPLIANCE_EVENT,
            "COMPLIANCE_REPORT_GENERATED",
            details={'period_days': days_back, 'total_events': total_events},
            severity=SeverityLevel.MEDIUM,
            compliance_tags=['SOX', 'GDPR', 'HIPAA']
        )
        
        return report
    
    def cleanup_old_records(self):
        """Clean up old audit records based on retention policy"""
        cutoff_date = (datetime.utcnow() - timedelta(days=self.retention_days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count records to be deleted
        cursor.execute('SELECT COUNT(*) FROM audit_events WHERE timestamp < ?', (cutoff_date,))
        records_to_delete = cursor.fetchone()[0]
        
        # Delete old records
        cursor.execute('DELETE FROM audit_events WHERE timestamp < ?', (cutoff_date,))
        conn.commit()
        conn.close()
        
        if records_to_delete > 0:
            self.log_event(
                EventType.SYSTEM_EVENT,
                "AUDIT_RECORDS_CLEANED",
                details={'records_deleted': records_to_delete, 'retention_days': self.retention_days},
                severity=SeverityLevel.MEDIUM
            )
        
        return records_to_delete

# Global audit trail manager instance
audit_manager = AuditTrailManager()

# Convenience functions for common audit events
def log_user_action(action: str, user_id: str = "anonymous", **kwargs):
    """Log user action with context"""
    return audit_manager.log_event(
        EventType.USER_ACTION,
        action,
        user_id=user_id,
        **kwargs
    )

def log_data_access(resource: str, user_id: str = "system", **kwargs):
    """Log data access event"""
    return audit_manager.log_event(
        EventType.DATA_ACCESS,
        f"DATA_ACCESSED: {resource}",
        user_id=user_id,
        resource=resource,
        **kwargs
    )

def log_api_call(api_endpoint: str, duration_ms: int, outcome: str = "SUCCESS", **kwargs):
    """Log API call event"""
    return audit_manager.log_event(
        EventType.API_CALL,
        f"API_CALL: {api_endpoint}",
        resource=api_endpoint,
        duration_ms=duration_ms,
        outcome=outcome,
        **kwargs
    )

def log_security_event(event: str, severity: SeverityLevel = SeverityLevel.HIGH, **kwargs):
    """Log security-related event"""
    return audit_manager.log_event(
        EventType.SECURITY_EVENT,
        event,
        severity=severity,
        compliance_tags=['SECURITY'],
        **kwargs
    )

def log_system_error(error: str, details: Dict[str, Any], **kwargs):
    """Log system error event"""
    return audit_manager.log_event(
        EventType.ERROR_EVENT,
        f"SYSTEM_ERROR: {error}",
        severity=SeverityLevel.HIGH,
        outcome="FAILURE",
        details=details,
        **kwargs
    )