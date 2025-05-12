import base64
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
# 1. Initialize client (ensure OPENAI_API_KEY is set in your environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 2. Prepare directories
os.makedirs("output_images", exist_ok=True)

# 3. Define your transformation prompt
prompt = """
Create a clean, professional marketing image of the attached LED product. Highlight a special launch offer with a bold discount tag that says: 'Just $25 – First Week Launch Offer'. 
Use a modern, minimalist design with good lighting to showcase the product, and ensure the discount stands out visually. 
White or light-colored background preferred.
Please make sure to keep the same led light as in the image.
Note Also keep the package just add this discount in the existing image 
mainly make sure the image is professional and modern.
And also keep the discount in red colour to match the package in the image
and also make sure the discount banner doesnt block the image
"""

# 4. Read input image as a file-like so MIME type is correct
input_path = "led.jpeg"
with open(input_path, "rb") as image_file:
    print("⏳ Generating image…")
    response = client.images.edit(
        model="gpt-image-1",
        image=image_file,
        prompt=prompt,
        size="1024x1024"
    )

# 5. Decode the Base64‑encoded result
b64_json = response.data[0].b64_json
new_img_data = base64.b64decode(b64_json)

# 6. Save the generated image
output_path = "output_images/transformed.jpg"
image = Image.open(BytesIO(new_img_data))
image.save(output_path, format="JPEG", quality=90, optimize=True)

print(f"✅ Image generation completed and saved to {output_path}")
