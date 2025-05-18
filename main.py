import os
import io
import base64
import logging
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image as PILImage

# OpenAI SDKa
import mlflow.langchain
from openai import OpenAI

# Google GenAI SDK
from google import genai
from google.genai import types


# LangChain / LangGraph
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# MLFLOW Imports
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.langchain.autolog()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MarketingAgent")

# Load env vars
load_dotenv()
logger.info("Loading API keys from .env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    logger.error("Missing OPENAI_API_KEY or GEMINI_API_KEY in environment")
    raise RuntimeError("API keys not found")

# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("Initialized OpenAI and Gemini clients")

# Ensure output dirs
os.makedirs("output_images", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)
logger.info("Ensured output directories exist")

# Tools
@tool(parse_docstring=True)
def gen_image(prompt: str, image_path: str) -> bytes:
    """
    Edits an image using OpenAI's image editing model.

    Args:
        prompt: The description of how to edit the image.
        image_path: Path to the input image file.

    Returns:
        The edited image data in bytes format.
    """
    with open(image_path, "rb") as f:
        resp = openai_client.images.edit(
            model="gpt-image-1",
            image=f,
            prompt=prompt,
            size="1024x1024"
        )
    return base64.b64decode(resp.data[0].b64_json)

@tool(parse_docstring=True)
def gen_video(prompt: str, image_path: str) -> bytes:
    """
    Generates a video from an image using Google's Gemini model.

    Args:
        prompt: The description of how to generate the video.
        image_path: Path to the input image file.

    Returns:
        The generated video data in bytes format (MP4).
    """
    pil = PILImage.open(image_path)
    buf = BytesIO()
    pil.save(buf, format=pil.format or "JPEG")
    img_bytes = buf.getvalue()

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
    while not op.done:
        op = gemini_client.operations.get(op)
    vid_ref = op.result.generated_videos[0].video
    resp = gemini_client.files.download(file=vid_ref)
    try:
        return resp.read()
    except Exception:
        return resp

# Registered tools
tools = [gen_image, gen_video]
tools_by_name = {t.name: t for t in tools}

# LLM setup
llm = ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# System prompt for main flow
tool_descriptions = (
    "- gen_image(prompt, image_path): Edits an existing image based on a description.\n"
    "- gen_video(prompt, image_path): Creates a short video based on an existing image and a description."
)
system_prompt = f"""
You are a marketing assistant specialized in generating visual content. You have access to the following tools:
{tool_descriptions}

RULES:
1. Analyze the user's request.
2. If the user explicitly asks for a 'video' or 'animation', use gen_video.
3. If the user explicitly asks for an 'image' or 'edit', use gen_image.
4. Only output a JSON tool call with 'name' and 'args'.
"""

# State definition
class State(TypedDict):
    messages: list
    image_path: str

# Rewrite node: rewrites prompts using image context and query
def rewrite_node(state: State):
    messages = state['messages']
    last = messages[-1]
    if isinstance(last, HumanMessage):
        raw_query = last.content.strip()
        if raw_query and not raw_query.startswith('{'):
            # Load current image and encode as base64
            with open(state['image_path'], 'rb') as img_f:
                img_data = img_f.read()
            b64 = base64.b64encode(img_data).decode('utf-8')
            # Create an image message for LLM
            image_msg = HumanMessage(content=[
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ])
            # Local system prompt for rewriting
            local_system = SystemMessage(
                content=(
                    "You are a marketing expert. First, examine the provided image context. "
                    "Then rewrite the user's brief into a detailed marketing prompt "
                    "that follows best practices in campaign planning, copywriting, and audience targeting."
                )
            )
            # Invoke LLM with image context and user query
            logger.info("Rewrite node: sending image context and query to LLM")
            out = llm.invoke([local_system, image_msg, HumanMessage(content=raw_query)])
            detailed = out.content.strip()
            logger.info("Rewrite node: got detailed prompt: %s", detailed)
            messages[-1] = HumanMessage(content=detailed)
    return {'messages': messages, 'image_path': state['image_path']}

# LLM node
def llm_node(state: State):
    out = llm_with_tools.invoke(state['messages'])
    return {'messages': state['messages'] + [out], 'image_path': state['image_path']}

# Action node
def action_node(state: State):
    messages = state['messages']
    last = messages[-1]
    if not getattr(last, 'tool_calls', None):
        return state
    call = last.tool_calls[0]
    fn = tools_by_name[call['name']]
    data = fn.invoke({
        'prompt': call['args'].get('prompt', ''),
        'image_path': state['image_path']
    })
    folder = 'output_images' if call['name']=='gen_image' else 'output_videos'
    ext = 'jpg' if call['name']=='gen_image' else 'mp4'
    out_path = os.path.join(folder, f"{call['name']}_out.{ext}")
    with open(out_path, 'wb') as f:
        f.write(data)
    messages.append(ToolMessage(content=f"{call['name']} result at: {out_path}", tool_call_id=call['id']))
    return {'messages': messages, 'image_path': out_path}

# Continue logic
def should_continue(state: State):
    has_tool = bool(getattr(state['messages'][-1], 'tool_calls', None))
    return 'Action' if has_tool else END

# Build graph
builder = StateGraph(State)
builder.add_node('Rewrite', rewrite_node)
builder.add_node('LLM', llm_node)
builder.add_node('Action', action_node)
builder.add_edge(START, 'Rewrite')
builder.add_edge('Rewrite', 'LLM')
builder.add_conditional_edges('LLM', should_continue, {'Action':'Action', END:END})
builder.add_edge('Action', 'LLM')
agent = builder.compile()

# CLI loop
if __name__ == '__main__':
    state = {
        'messages': [SystemMessage(content=system_prompt)],
        'image_path': 'led.jpeg'
    }
    print("Type 'exit' to quit.")
    while True:
        inp = input("Your prompt: ").strip()
        if inp.lower() in ('exit','quit'):
            break
        state['messages'].append(HumanMessage(content=inp))
        state = agent.invoke(state)
        print("✅ Done — check outputs.")