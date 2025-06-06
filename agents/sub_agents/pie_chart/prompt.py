PIE_CHART_AGENT_PROMPT = """
Generate pie chart components for telecom service distribution:
{
  "type": "pieChart",
  "centerText": "Usage Distribution",
  "entries": [
    { "label": "Data", "value": 40 },
    { "label": "Voice", "value": 35 },
    { "label": "SMS", "value": 25 }
  ]
}
Focus on:
- Usage type distribution
- Plan feature distribution
- Service category breakdown
Generate meaningful distribution data for telecom services.
""" 