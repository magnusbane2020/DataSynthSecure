# Overview

This project is a secure synthetic data analysis application that generates synthetic Salesforce opportunity data and scores it using AI. The application follows a three-step workflow: generating synthetic sales opportunity data, scoring opportunities using OpenAI's GPT model with MEDDPICC/BANT frameworks, and generating executive reports with business insights. The system emphasizes security by using only synthetic data, secure API key handling, and comprehensive logging with sensitive data filtering.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Application**: Uses Streamlit for the user interface with a multi-step workflow design
- **Step-based Navigation**: Three-phase analysis process with sidebar navigation for different stages
- **Security-first UI**: Includes security notices and disclaimers about synthetic data usage

## Backend Architecture
- **Modular Component Design**: Separated into distinct modules for data generation, AI scoring, report generation, and utilities
- **Object-oriented Classes**: Each major functionality is encapsulated in dedicated classes (SyntheticDataGenerator, OpportunityScorer, ExecutiveReportGenerator)
- **Secure Logging System**: Custom logging with security filters to prevent sensitive data from appearing in logs

## Data Generation Strategy
- **Faker Library Integration**: Uses Faker with fixed seeds for reproducible synthetic data
- **Fortune 500 Company Names**: Uses only publicly traded company names to avoid any real customer data
- **MEDDPICC Framework**: Implements sales qualification framework with partial completion for realistic scenarios
- **CSV Export**: Stores generated data locally as CSV files

## AI Scoring System
- **OpenAI GPT-5 Integration**: Uses the latest OpenAI model for opportunity scoring
- **MEDDPICC/BANT Frameworks**: Implements established sales qualification methodologies
- **JSON Response Format**: Structured AI responses for consistent parsing
- **Temperature Control**: Uses low temperature (0.3) for consistent scoring results

## Security Architecture
- **Environment Variable API Keys**: All API keys stored securely in environment variables
- **Synthetic Data Only**: No real customer or PII data processing
- **Security Logging Filters**: Custom filters to remove sensitive information from logs
- **Input Validation**: API key validation and secure data handling practices

# External Dependencies

## AI Services
- **OpenAI API**: Primary AI service for opportunity scoring using GPT-5 model
- **API Key Management**: Requires OPENAI_API_KEY environment variable

## Data Processing Libraries
- **Pandas**: Data manipulation and CSV handling
- **NumPy**: Numerical operations and random number generation
- **Faker**: Synthetic data generation with reproducible seeds

## Web Framework
- **Streamlit**: Web application framework for the user interface
- **Multi-page Architecture**: Step-based workflow implementation

## Utility Libraries
- **Logging**: Python's built-in logging with custom security filters
- **JSON**: Data serialization for AI API communication
- **DateTime**: Time-based operations and reporting
- **OS/Environment**: Secure environment variable handling

## File Storage
- **Local CSV Files**: Synthetic data stored in local CSV format
- **No External Storage**: All data remains local for security compliance