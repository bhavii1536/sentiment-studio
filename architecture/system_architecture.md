📐 System Architecture
Sentiment Analysis Studio: Real-Time Media Opinion Analysis Using ML
🔷 Overview

The Sentiment Analysis Studio is a modular, multilingual sentiment analysis system designed to analyze public opinions from datasets and social media platforms such as YouTube and Twitter. The system follows a layered architecture to ensure scalability, security, and maintainability.

🧩 Architectural Layers
1️⃣ User Interface Layer (Frontend)

Technology: Streamlit

Responsibilities:

Accepts user inputs:

CSV files

Product / topic names

Social media identifiers (YouTube, Twitter)

Displays:

Sentiment charts

Tabular insights

Downloadable results

Purpose:
Provides a simple, interactive web interface for users without requiring technical knowledge.

2️⃣ Input Routing Layer

Responsibilities:

Identifies input type (CSV, topic, link, channel)

Routes input to the appropriate processing pipeline

Benefit:
Allows multiple input types to be handled using a single unified dashboard.

3️⃣ Data Collection Layer

Sources:

CSV datasets (user uploaded)

YouTube Data API (comments & metadata)

Twitter public data (via snscrape)

Security:

API keys are stored securely using Streamlit Secrets

No sensitive credentials are committed to GitHub

4️⃣ Preprocessing & Language Detection Layer

Operations:

Text normalization

Emoji preservation

Language detection (English, Tamil, Hindi)

Tools Used:

langdetect

Python preprocessing utilities

Purpose:
Ensures correct routing of text to language-appropriate sentiment models.

5️⃣ Sentiment Analysis Layer (Core ML Engine)

Models Used:

RoBERTa (Twitter-trained) for English sentiment

XLM-RoBERTa for Tamil and Hindi sentiment

Output Classes:

Positive

Neutral

Negative

Advantage:
High accuracy using transformer-based deep learning models.

6️⃣ Aggregation & Visualization Layer

Functions:

Aggregates sentiment predictions

Generates:

Pie charts

Bar charts

Platform comparison graphs

Libraries:

Pandas

Matplotlib

7️⃣ Insight Generation Layer

Features:

Automatic sentiment summaries

Language-wise opinion trends

Platform-wise comparisons

Purpose:
Transforms raw sentiment predictions into meaningful insights.

8️⃣ Export & Reporting Layer

Capabilities:

Download analyzed CSV files

Supports offline analysis and reporting

🔄 Architecture Flow Diagram (Textual)
User Input
   ↓
Streamlit UI
   ↓
Input Routing
   ↓
Data Collection (CSV / YouTube / Twitter)
   ↓
Preprocessing & Language Detection
   ↓
Sentiment Analysis (RoBERTa / XLM-RoBERTa)
   ↓
Aggregation & Visualization
   ↓
Insights & Downloadable Reports

🔐 Security & Best Practices

API keys managed via Streamlit Secrets

No credentials stored in code or repository

Public data usage only

🚀 Future Enhancements

Aspect-based sentiment analysis

Emoji-based emotion scoring

Real-time trend monitoring

Creator-level analytics

📝 One-Line Summary

The system employs a modular architecture combining Streamlit, transformer-based NLP models, and secure API integrations to deliver real-time multilingual sentiment insights.
