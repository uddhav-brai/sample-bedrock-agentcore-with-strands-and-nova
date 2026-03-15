from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import calculator
import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

# Setup Nova Pro model ID based on AWS region
NOVA_PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
region = boto3.session.Session().region_name
if region.startswith("eu"):
    NOVA_PRO_MODEL_ID = "eu.amazon.nova-pro-v1:0"
elif region.startswith("ap"):
    NOVA_PRO_MODEL_ID = "apac.amazon.nova-pro-v1:0"

@tool
def weather(city: str) -> str:
    """Get weather information for a city
    Args:
        city: City or location name
    """
    return f"Weather for {city}: Sunny, 35°C"


# Create and test the comprehensive Strands Agent
agent = Agent(
    model=BedrockModel(model_id=NOVA_PRO_MODEL_ID),
    system_prompt = """You are a helpful assistant that provides concise responses.
                    """,
    tools=[weather, calculator],
)

@app.entrypoint
async def strands_agent_bedrock(payload, context):
    """
    Invoke the agent with a payload
    """
    print(f"Payload: {payload}")
    print(f"Context: {context}")
    user_input = payload.get("prompt", "No prompt found")
    response = agent(user_input)
    return response

    # Streaming Mode
    """
    stream = agent.stream_async(user_input)
    async for event in stream:
        if "data" in event:
            yield event
    """

if __name__ == "__main__":
    app.run()
