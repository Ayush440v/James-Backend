MAP_AGENT_PROMPT = """
Generate a map component showing a location:
{
  "type": "map",
  "latitude": 37.7749,
  "longitude": -122.4194
}
Use the serper_maps_tool tool to find the coordinates.
DO NOT MAKE UP COORDINATES.
""" 