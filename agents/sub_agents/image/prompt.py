IMAGE_AGENT_PROMPT = """
Generate one or more image components in JSON format. Each should include a valid image URL.
Return an array like:
[
  {"type": "image", "text": imageUrl}
]
Use the serper_image_tool tool to find the image URL. Use the imageURL as the text.
DO NOT MAKE UP IMAGE URLS.
""" 