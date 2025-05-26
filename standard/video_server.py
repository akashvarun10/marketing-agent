import os
import asyncio
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image as PILImage
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai import types

# Load env vars
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Ensure output dirs
os.makedirs("output_videos", exist_ok=True)

# Create MCP server
mcp = FastMCP("VideoGeneration")

@mcp.tool()
def gen_video(prompt: str, image_path: str) -> str:
    """
    Generates a video from an image using Google's Gemini model.

    Args:
        prompt: The description of how to generate the video.
        image_path: Path to the input image file.

    Returns:
        The path to the generated video file.
    """
    try:
        # Load and prepare image
        pil = PILImage.open(image_path)
        buf = BytesIO()
        pil.save(buf, format=pil.format or "JPEG")
        img_bytes = buf.getvalue()

        # Generate video using Gemini
        op = gemini_client.models.generate_videos(
            model="veo-2.0-generate-001",
            prompt=prompt,
            image=types.Image(image_bytes=img_bytes, mime_type="image/jpeg"),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                number_of_videos=1,
                duration_seconds=8,
            ),
        )
        
        # Wait for completion
        while not op.done:
            op = gemini_client.operations.get(op)
            
        # Download video
        vid_ref = op.result.generated_videos[0].video
        resp = gemini_client.files.download(file=vid_ref)
        
        video_data = resp.read() if hasattr(resp, 'read') else resp
        output_path = os.path.join("output_videos", "gen_video_out.mp4")
        
        with open(output_path, "wb") as f:
            f.write(video_data)
            
        return output_path
        
    except Exception as e:
        raise Exception(f"Video generation failed: {str(e)}")

if __name__ == "__main__":
    mcp.run(transport="stdio")
