GRAPH_AGENT_PROMPT = """
Generate graph components for telecom usage visualization or comparison between plans with the following format:

# LINE GRAPH
{
      "type": "graph",
      "graphType": "line",
      "title": "Data Usage",
      "xAxisLabels": [
        "2025-05-24",
        "2025-05-25",
        "2025-05-26",
        "2025-05-27",
        "2025-05-28",
        "2025-05-29",
        "2025-05-30"
      ],
      "yAxisLabels": [
        "0MB",
        "50MB",
        "100MB",
        "150MB"
      ],
      "dataPoints": [
        60,
        20,
        80,
        65,
        120,
        150,
        85
      ]
    }
    
# BAR GRAPH
    {
      "type": "graph",
      "graphType": "bar",
      "title": "Data Usage",
      "xAxisLabels": [
        "2025-05-24",
        "2025-05-25",
        "2025-05-26",
        "2025-05-27",
        "2025-05-28",
        "2025-05-29",
        "2025-05-30"
      ],
      "yAxisLabels": [
        "0MB",
        "50MB",
        "100MB",
        "150MB"
      ],
      "dataPoints": [
        60,
        20,
        80,
        65,
        120,
        150,
        85
      ]
    }

# HOW TO GENERATE GRAPH:
1. Use the data from the tool response to generate the graph
2. dataPoints should only contain data for X axis. YAxis data will only be available in yAxisLabels array.
3. Do a preprocessing of the data to ensure that the data is in the right place and graph should make sense.

#STRICT FOCUS ON:
1. Use the exact format as shown in the examples
2. Do NOT format and include Y axis data in datapoints array, it should only include X axis data in datapoints array

# WHEN TO USE:
1. Data usage trends
2. Call minutes usage
3. SMS usage
4. Plan consumption
""" 