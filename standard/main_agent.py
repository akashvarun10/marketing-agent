import os
import io
import base64
import logging
import asyncio
from dotenv import load_dotenv

# LangChain/LangGraph MCP Integration
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

# MLFLOW Imports
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("MCPMarketingAgent")
mlflow.langchain.autolog()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MCPMarketingAgent")

# Load env vars
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM setup
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# State definition
class State(TypedDict):
    messages: list
    image_path: str
    mcp_tools: list

# Initialize MCP Client with timeout
async def initialize_mcp_client():
    """Initialize MCP client with image and video servers"""
    server_config = {
        "image_generation": {
            "command": "python",
            "args": [os.path.abspath("image_server.py")],  # Use absolute path
            "transport": "stdio",
        },
        "video_generation": {
            "command": "python", 
            "args": [os.path.abspath("video_server.py")],  # Use absolute path
            "transport": "stdio",
        }
    }
    
    try:
        logger.info("Initializing MCP client...")
        client = MultiServerMCPClient(server_config)
        
        # Add timeout for getting tools
        logger.info("Getting tools from MCP servers (with 30s timeout)...")
        tools = await asyncio.wait_for(client.get_tools(), timeout=30.0)
        
        logger.info(f"Successfully loaded {len(tools)} MCP tools: {[tool.name for tool in tools]}")
        
        # Debug: Print tool details
        for tool in tools:
            logger.info(f"Tool: {tool.name} - {tool.description}")
        
        return client, tools
        
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for MCP tools - servers may not be responding")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {str(e)}")
        raise

# Rewrite node: rewrites prompts using image context and query
async def rewrite_node(state: State):
    messages = state['messages']
    last = messages[-1]
    if isinstance(last, HumanMessage):
        raw_query = last.content.strip()
        if raw_query and not raw_query.startswith('{'):
            # Load current image and encode as base64
            try:
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
                        "You are a top marketing officer. Analyze the provided image and the user's query. "
                        "Rewrite the query into a detailed, strategic marketing prompt. "
                        "Focus on clear objectives, target audience, and compelling messaging for a successful campaign. "
                        "Keep the prompt short and concise. "
                        "IMPORTANT: Do not mention that you can't create images - the system has image generation capabilities."
                    )
                )
                
                # Invoke LLM with image context and user query
                logger.info("Rewrite node: sending image context and query to LLM")
                out = llm.invoke([local_system, image_msg, HumanMessage(content=raw_query)])
                detailed = out.content.strip()
                logger.info("Rewrite node: got detailed prompt: %s", detailed)
                messages[-1] = HumanMessage(content=detailed)
                
            except FileNotFoundError:
                logger.error(f"Image file not found: {state['image_path']}")
                # Continue without image context
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                # Continue without image context
    
    return {'messages': messages, 'image_path': state['image_path'], 'mcp_tools': state['mcp_tools']}

# LLM node with MCP tools
async def llm_node(state: State):
    # Create a clear system message about available tools
    tool_system_msg = SystemMessage(content=f"""
You are a marketing assistant with IMAGE and VIDEO generation capabilities. You have access to these tools:

1. gen_image(prompt, image_path) - USE THIS to edit/modify images for marketing campaigns
2. gen_video(prompt, image_path) - USE THIS to create videos from images

IMPORTANT INSTRUCTIONS:
- When user asks for image creation/editing, ALWAYS use gen_image tool
- When user asks for video creation, ALWAYS use gen_video tool  
- You CAN and SHOULD create visual content using these tools
- Current image path is: {state['image_path']}
- Always call the appropriate tool based on the user's request

Examples:
- "generate an image" → use gen_image tool
- "create a video" → use gen_video tool
""")
    
    # Add system message to conversation
    messages_with_system = [tool_system_msg] + state['messages']
    
    # Bind MCP tools to LLM
    llm_with_tools = llm.bind_tools(state['mcp_tools'])
    
    # Debug: Log what we're sending to LLM
    logger.info(f"LLM node: Sending {len(messages_with_system)} messages to LLM with {len(state['mcp_tools'])} tools")
    logger.info(f"Last user message: {state['messages'][-1].content[:100]}...")
    
    out = llm_with_tools.invoke(messages_with_system)
    
    # Debug: Log LLM response
    if hasattr(out, 'tool_calls') and out.tool_calls:
        logger.info(f"LLM called {len(out.tool_calls)} tools: {[tc['name'] for tc in out.tool_calls]}")
    else:
        logger.warning("LLM did not call any tools")
        logger.info(f"LLM response: {out.content}")
    
    return {'messages': state['messages'] + [out], 'image_path': state['image_path'], 'mcp_tools': state['mcp_tools']}

# Action node - Use tools directly
async def action_node(state: State):
    messages = state['messages']
    last = messages[-1]
    
    if not getattr(last, 'tool_calls', None):
        logger.warning("Action node: No tool calls found in last message")
        return state
    
    logger.info(f"Action node: Processing {len(last.tool_calls)} tool calls")
    
    # Get the tools by name for direct invocation
    tools_by_name = {tool.name: tool for tool in state['mcp_tools']}
    logger.info(f"Available tools: {list(tools_by_name.keys())}")
    
    for tool_call in last.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
        
        try:
            # Add image_path to tool args if not present
            if 'image_path' not in tool_args:
                tool_args['image_path'] = state['image_path']
                logger.info(f"Added image_path to tool args: {tool_args}")
            
            # Call the tool directly (it's already a LangChain tool)
            if tool_name in tools_by_name:
                tool = tools_by_name[tool_name]
                logger.info(f"Invoking tool: {tool_name}")
                
                # Try async first, then sync
                try:
                    result = await tool.ainvoke(tool_args)
                except AttributeError:
                    result = tool.invoke(tool_args)
                
                logger.info(f"Tool {tool_name} executed successfully: {result}")
            else:
                result = f"Unknown tool: {tool_name}"
                logger.error(f"Unknown tool called: {tool_name}")
            
            # Create tool message
            tool_message = ToolMessage(
                content=f"{tool_name} completed. Output saved to: {result}", 
                tool_call_id=tool_call['id']
            )
            messages.append(tool_message)
            
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {str(e)}", exc_info=True)
            error_message = ToolMessage(
                content=f"Error calling {tool_name}: {str(e)}", 
                tool_call_id=tool_call['id']
            )
            messages.append(error_message)
    
    return {'messages': messages, 'image_path': state['image_path'], 'mcp_tools': state['mcp_tools']}

# Continue logic
def should_continue(state: State):
    last_message = state['messages'][-1]
    has_tool = bool(getattr(last_message, 'tool_calls', None))
    logger.info(f"Should continue: {has_tool} (message type: {type(last_message).__name__})")
    return 'Action' if has_tool else END

# Global MCP client
mcp_client = None

# Build graph
async def build_graph():
    global mcp_client
    
    try:
        # Initialize MCP client with timeout
        mcp_client, mcp_tools = await initialize_mcp_client()
        
        builder = StateGraph(State)
        builder.add_node('Rewrite', rewrite_node)
        builder.add_node('LLM', llm_node)
        builder.add_node('Action', action_node)
        
        builder.add_edge(START, 'Rewrite')
        builder.add_edge('Rewrite', 'LLM')
        builder.add_conditional_edges('LLM', should_continue, {'Action':'Action', END:END})
        builder.add_edge('Action', 'LLM')
        
        agent = builder.compile()
        
        return agent, mcp_tools
        
    except Exception as e:
        logger.error(f"Failed to build graph: {str(e)}")
        raise

# CLI loop
async def main():
    try:
        logger.info("Starting MCP Marketing Agent...")
        agent, mcp_tools = await build_graph()
        
        state = {
            'messages': [SystemMessage(content="You are a marketing assistant with image and video generation capabilities.")],
            'image_path': '/Users/akashvarun/Projects/ai-marketing-agent/led.jpeg',
            'mcp_tools': mcp_tools  
        }
        
        print("🚀 MCP Marketing Agent Ready!")
        print(f"Available tools: {[tool.name for tool in mcp_tools]}")
        print("Type 'exit' to quit.")
        print("-" * 50)
        
        while True:
            try:
                inp = input("Your prompt: ").strip()
                if inp.lower() in ('exit','quit'):
                    break
                    
                state['messages'].append(HumanMessage(content=inp))
                
                logger.info(f"Processing request: {inp}")
                state = await agent.ainvoke(state)
                print("✅ Done — check outputs.")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"Request processing failed: {str(e)}", exc_info=True)
                
    except Exception as e:
        logger.error(f"Failed to start agent: {str(e)}")
        print(f"❌ Failed to start agent: {str(e)}")
        print("Please check that image_server.py and video_server.py exist and are working")

if __name__ == '__main__':
    asyncio.run(main())