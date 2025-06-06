GRAPH_AGENT_PROMPT = """
Generate graph components for telecom usage visualization:
{
  "type": "graph",
  "graphType": "line",
  "title": "Data Usage Trend",
  "xAxisLabels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
  "yAxisLabels": ["0GB", "2GB", "4GB", "6GB", "8GB", "10GB"],
  "dataPoints": [
    { "x": 0, "y": 2 },
    { "x": 1, "y": 3 },
    { "x": 2, "y": 4 }
  ]
}
Focus on:
- Data usage trends
- Call minutes usage
- SMS usage
- Plan consumption
Generate realistic usage data visualization.
""" 