import requests
import json
import os
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def google_finance_search_tool(input_data: Dict[str, str]) -> Dict[str, Dict]:
    """
    Searches for financial information using SerpAPI's Google Finance integration.
    
    Args:
        input_data (Dict[str, str]): Dictionary containing:
            - query (str): The stock symbol or company name to search for (e.g., "GOOGL:NASDAQ", "AAPL:NASDAQ")
            - window (Optional[str]): Time range for the graph. Can be:
                - "1D" (1 Day, default)
                - "5D" (5 Days)
                - "1M" (1 Month)
                - "6M" (6 Months)
                - "YTD" (Year to Date)
                - "1Y" (1 Year)
                - "5Y" (5 Years)
                - "MAX" (Maximum)
            - hl (Optional[str]): Language code (e.g., "en" for English)
    
    Returns:
        Dict[str, Dict]: Processed financial information including:
            - summary: Basic stock information
            - graph: Price history data
            - key_events: Important events affecting the stock
            - news: Recent news articles
            - financials: Financial statements
    """
    try:
        # Extract parameters
        query = input_data.get("query")
        window = input_data.get("window", "1D")
        hl = input_data.get("hl", "en")
        
        if not query:
            return {"error": {"message": "Query parameter is required"}}
            
        # Get API key from environment
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            return {"error": {"message": "SERPAPI_KEY environment variable not set"}}
            
        # Construct API request
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_finance",
            "q": query,
            "window": window,
            "hl": hl,
            "api_key": api_key
        }
        
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Process and structure the response
        processed_data = {
            "summary": data.get("summary", {}),
            "graph": data.get("graph", []),
            "key_events": data.get("key_events", []),
            "news": data.get("news_results", []),
            "financials": data.get("financials", []),
            "markets": data.get("markets", {})
        }
        
        return processed_data
        
    except requests.exceptions.HTTPError as err:
        return {"error": {"message": f"HTTP Error: {err}"}}
    except Exception as e:
        return {"error": {"message": f"Error: {str(e)}"}}

# Create the tool instance
google_finance_tool = google_finance_search_tool 