import time
import os
import io
from dotenv import load_dotenv
from PIL import Image as PILImage
from google import genai
from google.genai import types

# Load your GEMINI_API_KEY from .env (ensure it's set there)
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = """
Create a 10-second promo video showcasing this LED product with smooth zoom-in
and rotation animations. Display the launch offer: 'Only $25 – First Week Launch!'
with eye-catching motion graphics. Use upbeat background music and a clean,
modern aesthetic.
"""

# Method 1: Use PIL to load the image and convert to bytes
image_path = "output_images/transformed.jpg"
pil_image = PILImage.open(image_path)

# Convert to bytes
image_bytes_io = io.BytesIO()
pil_image.save(image_bytes_io, format=pil_image.format or "JPEG")
image_bytes = image_bytes_io.getvalue()

# Pass the image correctly as a types.Image object
operation = client.models.generate_videos(
    model="veo-2.0-generate-001",
    prompt=prompt,
    image=types.Image(image_bytes=image_bytes, mime_type="image/jpeg"),
    config=types.GenerateVideosConfig(
        aspect_ratio="16:9",
        number_of_videos=1,
        duration_seconds=8,  # Specify duration (5-8 seconds)
    ),
)

# Poll until the operation completes
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)
    print(f"Operation status: {operation}")

# Download and save each generated video locally
for n, vid in enumerate(operation.result.generated_videos):
    fname = f"promo_video_{n}.mp4"
    client.files.download(file=vid.video)
    vid.video.save(fname)
    print("Saved:", fname)