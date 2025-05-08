import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI


class VisionManager:
    def __init__(self):
        """Initialize the VisionManager with OpenAI configuration."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get(
            "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")

    async def generate_and_save_image(self, prompt: str, output_path: str) -> str:
        """Generate an image using OpenAI API and save it to the specified path."""
        try:
            # Generate the image
            response = self.client.images.generate(
                model=self.model, prompt=prompt, n=1, size="1024x1024"
            )

            if not response.data:
                raise ValueError("No images generated")

            # Get the image data
            image_data = response.data[0].b64_json

            # Convert base64 to image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

            # Ensure the save directory exists
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            except OSError as e:
                raise ValueError(f"Failed to create directory for image: {str(e)}")

            # Save the image
            image.save(output_path)

        except Exception as e:
            raise ValueError(f"Failed to generate or save image: {str(e)}")

        return output_path
