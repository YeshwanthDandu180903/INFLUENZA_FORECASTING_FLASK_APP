
import os
import json
from utils.chatbot_logic import region_aware_response

# 1. Setup Mock Snapshot (simulating the dashboard)
mock_snapshot = {
    'region': 'Region 2',
    'ili_forecast_pct': 3.45,
    'severity': 'Medium',
    'trend': 'increasing',
    'dominant_virus': 'A (H1N1)',
    'jurisdictions': ['New Jersey', 'New York', 'Puerto Rico', 'U.S. Virgin Islands']
}

# 2. Test Queries
test_queries = [
    "What is the forecast for next week?",
    "Which states are in this region?",
    "Why is the risk medium?",
    "What is the dominant virus?"
]

print("--- RAG CHATBOT TEST ---")
for query in test_queries:
    print(f"\nUser: {query}")
    # We call the logic directly
    response = region_aware_response(query, mock_snapshot)
    print(f"Bot: {response}")
