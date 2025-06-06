IMAGE_GRID_AGENT_PROMPT = """
Create an 'imageGrid' component with 3-6 image URLs in JSON format:
{
  "type": "imageGrid",
  "items": ["url1", "url2", "url3"]
}
Use the serper_image_tool tool to find the image URL.
DO NOT MAKE UP IMAGE URLS.
""" 