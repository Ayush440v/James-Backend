from google.adk.tools import FunctionTool, ToolContext
import requests
import json
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# City to airport code mapping
AIRPORT_CODES = {
    "DELHI": "DEL",
    "MUMBAI": "BOM",
    "BANGALORE": "BLR",
    "CHENNAI": "MAA",
    "KOLKATA": "CCU",
    "HYDERABAD": "HYD",
    "DUBAI": "DXB",
    "ABU DHABI": "AUH",
    "SHARJAH": "SHJ",
    "LONDON": "LHR",
    "NEW YORK": "JFK",
    "LOS ANGELES": "LAX",
    "SINGAPORE": "SIN",
    "BANGKOK": "BKK",
    "HONG KONG": "HKG",
    "TOKYO": "HND",
    "SYDNEY": "SYD",
    "MELBOURNE": "MEL",
    "PARIS": "CDG",
    "FRANKFURT": "FRA",
    "AMSTERDAM": "AMS",
    "ROME": "FCO",
    "MADRID": "MAD",
    "ISTANBUL": "IST",
    "DOHA": "DOH",
    "RIYADH": "RUH",
    "JEDDAH": "JED",
    "CAIRO": "CAI",
    "JOHANNESBURG": "JNB",
    "NAIROBI": "NBO"
}

def get_airport_code(city: str) -> str:
    """Convert city name to airport code."""
    city_upper = city.upper()
    if city_upper in AIRPORT_CODES:
        return AIRPORT_CODES[city_upper]
    return city  # Return as is if not found in mapping

def google_flights_search_tool(input_data: dict, tool_context: ToolContext) -> dict:
    """
    Searches for flight information using SerpAPI's Google Flights integration.

    Args:
        input_data (dict): The input data containing search parameters:
            - origin (str): Origin city/airport (e.g., 'Delhi' or 'DEL')
            - destination (str): Destination city/airport (e.g., 'Dubai' or 'DXB')
            - departure_date (str): Departure date in YYYY-MM-DD format
            - return_date (str, optional): Return date in YYYY-MM-DD format for round trips
            - adults (int, optional): Number of adult passengers
            - children (int, optional): Number of child passengers
            - infants (int, optional): Number of infant passengers
            - cabin_class (str, optional): Cabin class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
            - currency (str, optional): Currency code (default: USD)
        tool_context (ToolContext): The context object providing access to session state.

    Returns:
        dict: The flight search results including:
            - best_flights: List of best flight options
            - other_flights: List of other flight options
            - price_insights: Price information and trends
            - airports: Airport information
    """
    logger.info(f"Received input data: {input_data}")

    try:
        # Extract search parameters from input_data
        origin = input_data.get("origin")
        destination = input_data.get("destination")
        departure_date = input_data.get("departure_date")
        return_date = input_data.get("return_date")
        adults = input_data.get("adults", 1)
        children = input_data.get("children", 0)
        infants = input_data.get("infants", 0)
        cabin_class = input_data.get("cabin_class", "ECONOMY")
        currency = input_data.get("currency", "USD")

        # Validate required parameters
        if not all([origin, destination, departure_date]):
            raise ValueError("Missing required parameters: origin, destination, and departure_date are required")

        # Convert city names to airport codes
        origin_code = get_airport_code(origin)
        destination_code = get_airport_code(destination)
        logger.info(f"Converted city codes - Origin: {origin} -> {origin_code}, Destination: {destination} -> {destination_code}")

        # Format dates
        try:
            departure_date_obj = datetime.strptime(departure_date, "%Y-%m-%d")
            if return_date:
                return_date_obj = datetime.strptime(return_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date format. Please use YYYY-MM-DD format")

        # Get SerpAPI key from environment
        api_key = os.getenv("SERPAPI_KEY")
        logger.info(f"API Key found: {'Yes' if api_key else 'No'}")
        
        if not api_key:
            logger.error("SERPAPI_KEY not found in environment variables")
            return {
                "error": "API key not configured",
                "message": "Please set the SERPAPI_KEY environment variable",
                "flights": [],
                "prices": {},
                "airlines": {},
                "airports": {}
            }

        # Construct the API request
        url = "https://serpapi.com/search"
        
        # Map cabin class to SerpAPI format
        cabin_class_map = {
            "ECONOMY": "1",
            "PREMIUM_ECONOMY": "2",
            "BUSINESS": "3",
            "FIRST": "4"
        }

        # Prepare parameters
        params = {
            "engine": "google_flights",
            "api_key": api_key,
            "departure_id": origin_code,
            "arrival_id": destination_code,
            "outbound_date": departure_date,
            "currency": currency,
            "hl": "en",  # English language
            "type": "1" if return_date else "2",  # 1 for round trip, 2 for one way
            "adults": adults,
            "children": children,
            "infants_in_seat": infants,
            "travel_class": cabin_class_map.get(cabin_class.upper(), "1")
        }

        # Add return date if provided
        if return_date:
            params["return_date"] = return_date

        logger.info(f"Making request to SerpAPI with params: {params}")

        # Make the API request
        response = requests.get(url, params=params)
        logger.info(f"SerpAPI Response Status: {response.status_code}")
        
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        logger.info(f"SerpAPI Response: {json.dumps(data, indent=2)}")
        
        # Check for errors in the response
        if "error" in data:
            logger.error(f"SerpAPI error: {data['error']}")
            return {
                "error": data["error"],
                "flights": [],
                "prices": {},
                "airlines": {},
                "airports": {}
            }

        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to SerpAPI: {str(e)}")
        return {
            "error": str(e),
            "flights": [],
            "prices": {},
            "airlines": {},
            "airports": {}
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            "error": str(e),
            "flights": [],
            "prices": {},
            "airlines": {},
            "airports": {}
        }

# Initialize the FunctionTool
google_flights_tool = FunctionTool(func=google_flights_search_tool) 