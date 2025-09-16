import logging
import os
import sys
import re
import pandas as pd
from logging.handlers import RotatingFileHandler

class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs"""
    
    def filter(self, record):
        # Remove API keys, tokens, or other sensitive data from log messages
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            # Basic filtering - can be expanded
            sensitive_patterns = ['api_key', 'token', 'password', 'secret']
            for pattern in sensitive_patterns:
                if pattern.lower() in msg.lower():
                    record.msg = '[FILTERED: Sensitive data removed]'
                    break
        return True

def setup_logging():
    """Setup secure logging configuration"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add security filter
    security_filter = SecurityFilter()
    console_handler.addFilter(security_filter)
    
    logger.addHandler(console_handler)
    
    # File handler (optional)
    try:
        file_handler = RotatingFileHandler(
            'app.log', maxBytes=1024*1024, backupCount=3
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        file_handler.addFilter(security_filter)
        logger.addHandler(file_handler)
    except Exception:
        # If file logging fails, continue with console only
        pass
    
    logger.info("Secure logging initialized")

def validate_api_key():
    """Validate OpenAI API key without logging its value"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False
    
    # Basic validation without exposing the key
    if len(api_key) < 20:  # OpenAI keys are longer
        return False
    
    if not api_key.startswith(('sk-', 'sk-proj-')):  # OpenAI key format
        return False
    
    return True

def sanitize_data_for_logging(data):
    """Sanitize data before logging to remove PII"""
    if isinstance(data, dict):
        sanitized = {}
        sensitive_fields = ['email', 'phone', 'address', 'ssn', 'api_key', 'token']
        
        for key, value in data.items():
            if any(field in key.lower() for field in sensitive_fields):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = value
        return sanitized
    
    return data

def validate_synthetic_data(df):
    """Validate that generated data contains no real PII"""
    # Check for common PII patterns
    pii_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone pattern
    ]
    
    # This is a basic check - in production, more sophisticated validation would be needed
    validation_passed = True
    
    # Check string columns for PII patterns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        for _, value in df[col].items():
            if pd.isna(value):
                continue
            str_value = str(value)
            for pattern in pii_patterns:
                if re.search(pattern, str_value):
                    validation_passed = False
                    logging.warning(f"Potential PII detected in column {col}")
                    break
    
    return validation_passed

# Additional utility functions for security
def mask_sensitive_string(text, mask_char='*', visible_chars=4):
    """Mask sensitive strings showing only first/last few characters"""
    if not text or len(text) <= visible_chars * 2:
        return mask_char * len(text) if text else ''
    
    return text[:visible_chars] + mask_char * (len(text) - visible_chars * 2) + text[-visible_chars:]

def generate_audit_log_entry(action, user_id=None, additional_info=None):
    """Generate structured audit log entry"""
    import json
    from datetime import datetime
    
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'user_id': user_id or 'system',
        'additional_info': additional_info or {}
    }
    
    return json.dumps(log_entry)
