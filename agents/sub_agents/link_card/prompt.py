LINK_CARD_AGENT_PROMPT = """
Generate a link card component with a valid URL:
{
  "type": "linkCard",
  "text": imageUrl
}

Use the serper_image_tool tool to find the image URL. Use the imageURL as the text.
DO NOT MAKE UP URLS.
""" 