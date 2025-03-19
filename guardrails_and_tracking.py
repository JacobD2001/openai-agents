import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent, InputGuardrail, GuardrailFunctionOutput, Runner, 
    FileSearchTool, WebSearchTool, RunConfig, InputGuardrailTripwireTriggered,
    RunContextWrapper
)

load_dotenv()


# ==========================================================================
# 1. DEFINING TOOLS
# ==========================================================================

# Create a FileSearchTool for document-based knowledge agent
document_file_search = FileSearchTool(
    vector_store_ids=[os.getenv("VECTOR_STORE_ID")], 
    max_num_results=3,
)

# Create a WebSearchTool for the web search agent
web_search = WebSearchTool()

# ==========================================================================
# 2. CREATING SPECIALIST AGENTS
# ==========================================================================
# Document knowledge agent that uses RAG with FileSearchTool
rag_agent = Agent(
    name="Brainli RAG Agent",
    handoff_description="Specialist agent for answering questions about Brainli based on the documents in the knowledge base",
    instructions="""You are a Brainli RAG agent who provides accurate information about Brainli based on the documents in the knowledge base.
Your primary purpose is to answer business-related questions using the file search tool to retrieve relevant information from documents.
Always cite your sources when retrieving information.
Be precise and factual in your responses, and acknowledge when information might not be available in the documents.
Use the file search tool to look for relevant information before answering.""",
    tools=[document_file_search],
)

# Web search agent for retrieving up-to-date information
search_agent = Agent(
    name="Brainli Web Search Agent",
    handoff_description="Specialist agent for retrieving up-to-date business information from the web",
    instructions="""You are a Brainli Web Search agent who provides up-to-date business information from the web.
Your primary role is to search the internet for current business information, market trends, and industry news.
Always use the web search tool to find current and accurate business information before responding.
Clearly indicate when information comes from web searches and cite your sources.
If search results are limited, acknowledge that and explain what you were able to find.""",
    tools=[web_search],
)

# ==========================================================================
# 3. IMPLEMENTING GUARDRAILS
# ==========================================================================

# Define the classification model for the guardrail
class BusinessQueryClassification(BaseModel):
    """Classification of the user query to determine if it's business-related and which agent to route to."""
    is_business_related: bool
    reasoning: str

# Guardrail agent to determine if the query is valid and classify it
guardrail_agent = Agent(
    name="Brainli Guardrail Agent",
    instructions="""Determine if the user query is business-related and appropriate to answer.
A business-related query pertains to corporate operations, market trends, industry news, 
business strategies, Brainli products/services, or professional workplace matters.
Classify the query to help with routing to the appropriate specialist agent.""",
    output_type=BusinessQueryClassification,
)

# Guardrail function that runs before the main agent
async def business_query_guardrail(ctx: RunContextWrapper, agent: Agent, input_data: str):
    """Guardrail function to classify queries and ensure they're business-related."""
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(BusinessQueryClassification)
    
    # Trigger the tripwire if the query is not business-related
    if not final_output.is_business_related:
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=True,
        )
    
    # Not triggering the tripwire if the query is business-related
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=False,
    )

# ==========================================================================
# 4. DELEGATION WITH GUARDRAILS AND TRACKING
# ==========================================================================
# Enhanced delegation agent with guardrails and tracking
delegation_agent = Agent(
    name="Brainli Delegation Agent",
    instructions="""You are the Brainli Delegation Agent who receives user queries and determines which specialist to route them to.
For questions about Brainli, its services, or business information that might be in the knowledge base, delegate to the Brainli RAG Agent.
For questions requiring current business information, market trends, or real-time business data, delegate to the Brainli Web Search Agent.
Analyze the query carefully to make the appropriate routing decision, focusing only on business-related inquiries.""",
    handoffs=[rag_agent, search_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=business_query_guardrail),
    ],
)

# ==========================================================================
# 5. EXECUTION WITH TRACING
# ==========================================================================
async def main():
    """Run the demo with examples demonstrating guardrails and tracing."""
    # Configure tracing for the workflow
    run_config = RunConfig(
        workflow_name="Brainli Business Assistant with Guardrails",
    )
    
    # Test with a business document-related question
    document_query = "What services does Brainli offer to help with business analytics?"
    
    result = await Runner.run(
        delegation_agent, 
        document_query,
        run_config=run_config
    )
    
    print(result.final_output)
    
    # Test with a business web search question
    web_query = "What are the latest business intelligence trends in 2024?"
    
    # Create a new run config for distinct tracing
    web_run_config = RunConfig(
        workflow_name="Web Search configuration",
    )
    
    result = await Runner.run(
        delegation_agent, 
        web_query,
        run_config=web_run_config
    )
    
    print(result.final_output)
    
    # Test with a non-business query (testing guardrail)
    non_business_query = "What's the recipe for chocolate chip cookies?"
    
    non_business_run_config = RunConfig(
        workflow_name="Non business workflow",
    )
    
    try:
        await Runner.run(
            delegation_agent, 
            non_business_query,
            run_config=non_business_run_config
        )
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail triggered")

if __name__ == "__main__":
    asyncio.run(main()) 