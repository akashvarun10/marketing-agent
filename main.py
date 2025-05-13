# import os
# import io
# import base64
# import logging
# from io import BytesIO
# from dotenv import load_dotenv
# from PIL import Image as PILImage

# # OpenAI SDKa
# import mlflow.langchain
# from openai import OpenAI

# # Google GenAI SDK
# from google import genai
# from google.genai import types

# # LangChain / LangGraph
# from langchain_core.tools import tool
# from typing_extensions import TypedDict
# from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END

# #MLFLOW Imports 
# import mlflow

# mlflow.set_tracking_uri("http://localhost:5050")
# mlflow.langchain.autolog()

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger = logging.getLogger("MarketingAgent")


# load_dotenv()
# logger.info("Loading API keys from .env")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not OPENAI_API_KEY or not GEMINI_API_KEY:
#     logger.error("Missing OPENAI_API_KEY or GEMINI_API_KEY in environment")
#     raise RuntimeError("API keys not found")

# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# logger.info("Initialized OpenAI and Gemini clients")


# os.makedirs("output_images", exist_ok=True)
# os.makedirs("output_videos", exist_ok=True)
# logger.info("Ensured output_images/ and output_videos/ exist")

# @tool(parse_docstring=True)
# def gen_image(prompt: str, image_path: str) -> bytes:
#     """Edits an image using OpenAI's image editing model.

#     Args:
#         prompt (str): The prompt describing how to edit the image.
#         image_path (str): Path to the input image file.

#     Returns:
#         bytes: The edited image data in bytes format.
#     """
#     logger.info("gen_image: reading %s", image_path)
#     with open(image_path, "rb") as f:
#         resp = openai_client.images.edit(
#             model="gpt-image-1",
#             image=f,
#             prompt=prompt,
#             size="1024x1024"
#         )
#     img_data = base64.b64decode(resp.data[0].b64_json)
#     logger.info("gen_image: decoded image (%d bytes)", len(img_data))
#     return img_data


# @tool(parse_docstring=True)
# def gen_video(prompt: str, image_path: str) -> bytes:
#     """Generates a video from an image using Google's Gemini model.

#     Args:
#         prompt (str): The prompt describing how to generate the video.
#         image_path (str): Path to the input image file.

#     Returns:
#         bytes: The generated video data in bytes format.
#     """
#     logger.info("gen_video: opening image %s", image_path)
#     pil = PILImage.open(image_path)
#     buf = BytesIO()
#     pil.save(buf, format=pil.format or "JPEG")
#     img_bytes = buf.getvalue()

#     op = gemini_client.models.generate_videos(
#         model="veo-2.0-generate-001",
#         prompt=prompt,
#         image=types.Image(image_bytes=img_bytes, mime_type="image/jpeg"),
#         config=types.GenerateVideosConfig(
#             aspect_ratio="16:9",
#             number_of_videos=1,
#             duration_seconds=8,
#         ),
#     )

#     # Poll until ready
#     logger.info("gen_video: polling for operation completion")
#     while not op.done:
#         logger.debug("gen_video: operation not done → sleeping 20s")
#         op = gemini_client.operations.get(op)
#     vid_ref = op.result.generated_videos[0].video
#     logger.info("gen_video: downloading video from reference %s", vid_ref)

#     # Download video bytes
#     resp = gemini_client.files.download(file=vid_ref)
#     try:
#         data = resp.read()
#     except Exception:
#         data = resp
#     logger.info("gen_video: downloaded video (%d bytes)", len(data))
#     return data

# logger.info("Binding tools to LLM")
# tools = [gen_image, gen_video]
# tools_by_name = {t.name: t for t in tools}

# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
# llm_with_tools = llm.bind_tools(tools)

# class State(TypedDict):
#     messages: list
#     image_path: str


# def llm_node(state: State):
#     logger.info("LLM node: sending %d messages to LLM", len(state["messages"]))
#     out = llm_with_tools.invoke(state["messages"])
#     logger.info("LLM node: received response with %d tool_calls", len(out.tool_calls))
#     return {"messages": state["messages"] + [out], "image_path": state["image_path"]}

# def action_node(state: State):
#     """Execute a single tool call from the LLM's response."""
#     messages = state["messages"]
#     last_message = messages[-1]
#     current_path = state["image_path"]
    
#     # Skip if no tool calls
#     if not last_message.tool_calls:
#         return {"messages": messages, "image_path": current_path}
    
#     # Only process the FIRST tool call - preventing chaining by design
#     call = last_message.tool_calls[0]
#     name = call["name"]
    
#     logger.info(f"Action node: Processing only first tool call: {name}")
    
#     # Execute the tool
#     fn = tools_by_name[name]
#     data = fn.invoke({
#         "prompt": call["args"].get("prompt", ""),
#         "image_path": current_path
#     })

#     # Process and save result
#     if name == "gen_image":
#         folder, ext = "output_images", "jpg"
#         out_path = os.path.join(folder, f"{name}_out.{ext}")
#         current_path = out_path  # Update path only for image
#     else:
#         folder, ext = "output_videos", "mp4"
#         out_path = os.path.join(folder, f"{name}_out.{ext}")
    
#     logger.info(f"Action node: saving {name} → {out_path}")
#     with open(out_path, "wb") as f:
#         f.write(data)

#     # Add tool response
#     messages.append(
#         ToolMessage(
#             content=f"{name} result stored at: {out_path}",
#             tool_call_id=call["id"]
#         )
#     )
    
#     # Handle any remaining tool calls with response messages
#     for skipped_call in last_message.tool_calls[1:]:
#         logger.info(f"Action node: Skipping additional tool call: {skipped_call['name']}")
#         messages.append(
#             ToolMessage(
#                 content=f"Skipped {skipped_call['name']} to prevent tool chaining",
#                 tool_call_id=skipped_call["id"]
#             )
#         )
    
#     return {"messages": messages, "image_path": current_path}

# def should_continue(state: State):
#     has_calls = bool(state["messages"][-1].tool_calls)
#     logger.info("should_continue: last message had tool_calls=%s", has_calls)
#     return "Action" if has_calls else END

# logger.info("Building StateGraph")
# builder = StateGraph(State)
# builder.add_node("LLM", llm_node)
# builder.add_node("Action", action_node)
# builder.add_edge(START, "LLM")
# builder.add_conditional_edges("LLM", should_continue, {"Action": "Action", END: END})
# builder.add_edge("Action", "LLM")

# agent = builder.compile()
# logger.info("Agent compiled successfully")

# if __name__ == "__main__":
#     system_prompt = (
#         "You are a marketing assistant. You have two tools:\n"
#         "- gen_image(prompt, image_path): edits an image.\n"
#         "- gen_video(prompt, image_path): makes a video from an image.\n"
#         "STRICT RULE: Only use the EXACT tool type that the user explicitly requests.\n"
#         "If user mentions 'image' or 'picture', use ONLY gen_image.\n"
#         "If user mentions 'video', use ONLY gen_video.\n"
#         "NEVER chain tools or use both tools unless the user explicitly asks for both.\n"
#         "DO NOT generate a video after generating an image unless specifically asked.\n"
#         "Always reply with a tool call with fields `name` and `args`."
#     )
#     state = {
#         "messages": [SystemMessage(content=system_prompt)],
#         "image_path": "led.jpeg"
#     }

#     print("Type 'exit' or 'quit' to stop.")
#     while True:
#         user_input = input("\nYour prompt: ").strip()
#         if user_input.lower() in ("exit", "quit"):
#             print("Goodbye!")
#             break

#         state["messages"].append(HumanMessage(content=user_input))
#         state = agent.invoke(state)
#         print("✅ Done — check output_images/ or output_videos/")

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

mlflow.set_tracking_uri("http://localhost:5050")
mlflow.langchain.autolog()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MarketingAgent")

load_dotenv()
logger.info("Loading API keys from .env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not OPENAI_API_KEY or not GEMINI_API_KEY:
    logger.error("Missing OPENAI_API_KEY or GEMINI_API_KEY in environment")
    raise RuntimeError("API keys not found")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("Initialized OpenAI and Gemini clients")

os.makedirs("output_images", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)
logger.info("Ensured output_images/ and output_videos/ exist")

@tool(parse_docstring=True)
def gen_image(prompt: str, image_path: str) -> bytes:
    """Edits an image using OpenAI's image editing model.

    Args:
        prompt (str): The prompt describing how to edit the image.
        image_path (str): Path to the input image file.

    Returns:
        bytes: The edited image data in bytes format.
    """
    logger.info("gen_image: reading %s", image_path)
    with open(image_path, "rb") as f:
        resp = openai_client.images.edit(
            model="gpt-image-1",
            image=f,
            prompt=prompt,
            size="1024x1024"
        )
    img_data = base64.b64decode(resp.data[0].b64_json)
    logger.info("gen_image: decoded image (%d bytes)", len(img_data))
    return img_data

@tool(parse_docstring=True)
def gen_video(prompt: str, image_path: str) -> bytes:
    """Generates a video from an image using Google's Gemini model.

    Args:
        prompt (str): The prompt describing how to generate the video.
        image_path (str): Path to the input image file.

    Returns:
        bytes: The generated video data in bytes format.
    """
    logger.info("gen_video: opening image %s", image_path)
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

    # Poll until ready
    logger.info("gen_video: polling for operation completion")
    while not op.done:
        logger.debug("gen_video: operation not done → sleeping 20s")
        op = gemini_client.operations.get(op)
    vid_ref = op.result.generated_videos[0].video
    logger.info("gen_video: downloading video from reference %s", vid_ref)

    # Download video bytes
    resp = gemini_client.files.download(file=vid_ref)
    try:
        data = resp.read()
    except Exception:
        data = resp
    logger.info("gen_video: downloaded video (%d bytes)", len(data))
    return data

logger.info("Binding tools to LLM")
tools = [gen_image, gen_video]
tools_by_name = {t.name: t for t in tools}

# ——— LLM setup and system prompt defined here ———
llm = ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# system_prompt = (
#     "You are a marketing assistant. You have two tools:\n"
#     "- gen_image(prompt, image_path): edits an image.\n"
#     "- gen_video(prompt, image_path): makes a video from an image.\n"
#     "STRICT RULE: Only use the EXACT tool type that the user explicitly requests.\n"
#     "If user mentions 'image' or 'picture', use ONLY gen_image.\n"
#     "If user mentions 'video', use ONLY gen_video.\n"
#     "NEVER chain tools or use both tools unless the user explicitly asks for both.\n"
#     "DO NOT generate a video after generating an image unless specifically asked.\n"
#     "Always reply with a tool call with fields `name` and `args`."
# )

system_prompt = (
    "You are a marketing assistant specialized in generating visual content. You have access to the following tools:\n"
    "- gen_image(prompt, image_path): Edits an existing image based on a description.\n"
    "- gen_video(prompt, image_path): Creates a short video based on an existing image and a description.\n\n"
    "Your primary goal is to fulfill the user's request using the appropriate tool.\n"
    "RULES:\n"
    "1. Carefully analyze the user's request.\n"
    "2. If the user explicitly asks for a 'video', 'animation', or 'promo video', you MUST use the `gen_video` tool.\n"
    "3. If the user explicitly asks for an 'image', 'picture', or 'edit', you MUST use the `gen_image` tool.\n"
    "4. Extract the user's description of the desired content as the `prompt` argument for the tool.\n"
    "5. Use the provided `image_path` for the tool call.\n"
    "6. Only use one tool per user request unless explicitly asked to perform a sequence.\n"
    "7. If the request clearly maps to one of the tools, respond ONLY with the corresponding tool call (JSON format with 'name' and 'args'). Do not add conversational text before or after the tool call."
    # Removed the "Always reply with a tool call..." line and rephrased the objective.
)
# ————————————————————————————————

class State(TypedDict):
    messages: list
    image_path: str

def llm_node(state: State):
    logger.info("LLM node: sending %d messages to LLM", len(state["messages"]))
    out = llm_with_tools.invoke(state["messages"])
    logger.info("LLM node: received response with %d tool_calls", len(out.tool_calls))
    return {"messages": state["messages"] + [out], "image_path": state["image_path"]}

def action_node(state: State):
    """Execute a single tool call from the LLM's response."""
    messages = state["messages"]
    last_message = messages[-1]
    current_path = state["image_path"]
    
    # Skip if no tool calls
    if not last_message.tool_calls:
        return {"messages": messages, "image_path": current_path}
    
    # Only process the FIRST tool call - preventing chaining by design
    call = last_message.tool_calls[0]
    name = call["name"]
    
    logger.info(f"Action node: Processing only first tool call: {name}")
    
    # Execute the tool
    fn = tools_by_name[name]
    data = fn.invoke({
        "prompt": call["args"].get("prompt", ""),
        "image_path": current_path
    })

    # Process and save result
    if name == "gen_image":
        folder, ext = "output_images", "jpg"
        out_path = os.path.join(folder, f"{name}_out.{ext}")
        current_path = out_path  # Update path only for image
    else:
        folder, ext = "output_videos", "mp4"
        out_path = os.path.join(folder, f"{name}_out.{ext}")
    
    logger.info(f"Action node: saving {name} → {out_path}")
    with open(out_path, "wb") as f:
        f.write(data)

    # Add tool response
    messages.append(
        ToolMessage(
            content=f"{name} result stored at: {out_path}",
            tool_call_id=call["id"]
        )
    )
    
    # Handle any remaining tool calls with response messages
    for skipped_call in last_message.tool_calls[1:]:
        logger.info(f"Action node: Skipping additional tool call: {skipped_call['name']}")
        messages.append(
            ToolMessage(
                content=f"Skipped {skipped_call['name']} to prevent tool chaining",
                tool_call_id=skipped_call["id"]
            )
        )
    
    return {"messages": messages, "image_path": current_path}

def should_continue(state: State):
    has_calls = bool(state["messages"][-1].tool_calls)
    logger.info("should_continue: last message had tool_calls=%s", has_calls)
    return "Action" if has_calls else END

logger.info("Building StateGraph")
builder = StateGraph(State)
builder.add_node("LLM", llm_node)
builder.add_node("Action", action_node)
builder.add_edge(START, "LLM")
builder.add_conditional_edges("LLM", should_continue, {"Action": "Action", END: END})
builder.add_edge("Action", "LLM")

agent = builder.compile()
logger.info("Agent compiled successfully")

if __name__ == "__main__":
    # Initialize the conversation state using the prompt defined above
    state = {
        "messages": [SystemMessage(content=system_prompt)],
        "image_path": "led.jpeg"
    }

    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYour prompt: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = agent.invoke(state)
        print("✅ Done — check output_images/ or output_videos/")
