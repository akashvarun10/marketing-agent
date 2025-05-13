# AI Marketing Agent

An AI-powered marketing assistant that can edit images and generate videos using OpenAI and Google's Gemini models.

## Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- MLflow server running on localhost:5050

## Setup

1. Install uv if you haven't already:
```bash
pip install uv
```

2. Install dependencies:
```bash
uv sync
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Running the Application

1. Start the MLflow server in a separate terminal:
```bash
uv run mlflow server --host 0.0.0.0 --port 5050
```

2. Run the application:
```bash
uv run main.py
```

3. The application will start and wait for your input. You can:
   - Edit images by providing prompts mentioning "image" or "picture"
   - Generate videos by providing prompts mentioning "video"
   - Type 'exit' or 'quit' to stop the application

## Output

- Edited images are saved in the `output_images/` directory
- Generated videos are saved in the `output_videos/` directory
- MLflow tracking data is available at http://localhost:5050

## Project Structure

- `main.py`: Main application file
- `requirements.txt`: Project dependencies
- `.env`: Environment variables for API keys
- `output_images/`: Directory for generated images
- `output_videos/`: Directory for generated videos
