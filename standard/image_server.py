import os
import base64
import logging
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image as PILImage
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure output dirs
os.makedirs("output_images", exist_ok=True)

# Create MCP server
mcp = FastMCP("ImageGeneration")

@mcp.tool()
def gen_image(prompt: str, image_path: str) -> str:
    """
    Edits an image using OpenAI's image editing model.

    Args:
        prompt: The description of how to edit the image.
        image_path: Path to the input image file.

    Returns:
        The path to the edited image file.
    """
    try:
        with open(image_path, "rb") as f:
            resp = openai_client.images.edit(
                model="gpt-image-1",
                image=f,
                prompt=prompt,
                size="1024x1024"
            )
        
        # Decode and save the image
        image_data = base64.b64decode(resp.data[0].b64_json)
        output_path = os.path.join("output_images", "gen_image_out.jpg")
        
        with open(output_path, "wb") as f:
            f.write(image_data)
            
        return output_path
        
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    mcp.run(transport="stdio")