import asyncio
from pydantic import BaseModel
from agents import (
    Agent, InputGuardrail, GuardrailFunctionOutput, Runner, 
    FileSearchTool, WebSearchTool, RunConfig
)

# ==========================================================================
# GUARDRAILS AND TRACKING TUTORIAL
# ==========================================================================
# This tutorial demonstrates advanced concepts:
# 1. Creating guardrails to validate and classify user inputs
# 2. Implementing tracing for monitoring agent workflows
# 3. Applying these features to a multi-agent system

# ==========================================================================
# 1. DEFINING TOOLS
# ==========================================================================
# Create a FileSearchTool for document-based knowledge agent
document_file_search = FileSearchTool(
    vector_store_ids=["vs_67d92ca1f8988191a68174aa06c76df3"], 
    max_num_results=3,
)

# Create a WebSearchTool for the web search agent
web_search = WebSearchTool()

# ==========================================================================
# 2. CREATING SPECIALIST AGENTS
# ==========================================================================
# Document knowledge agent that uses RAG with FileSearchTool
rag_agent = Agent(
    name="Document Knowledge Agent",
    handoff_description="Specialist agent for answering questions using document knowledge",
    instructions="""You are a document knowledge agent who provides accurate information from documents in the knowledge base.
Your primary purpose is to answer questions using the file search tool to retrieve relevant information from documents.
Always cite your sources when retrieving information.
Be precise and factual in your responses, and acknowledge when information might not be available in the documents.
Use the file search tool to look for relevant information before answering.""",
    tools=[document_file_search],
)

# Web search agent for retrieving up-to-date information
search_agent = Agent(
    name="Web Search Agent",
    handoff_description="Specialist agent for retrieving up-to-date information from the web",
    instructions="""You provide up-to-date information from the web on a wide range of topics.
Your primary role is to search the internet for current information that may not be available in static documents.
Always use the web search tool to find current and accurate information before responding.
Clearly indicate when information comes from web searches and cite your sources.
If search results are limited, acknowledge that and explain what you were able to find.""",
    tools=[web_search],
)

# ==========================================================================
# 3. IMPLEMENTING GUARDRAILS
# ==========================================================================
# Define the classification model for the guardrail
class TopicClassification(BaseModel):
    """Classification of the user query to determine appropriate agent routing."""
    is_document_related: bool
    requires_web_search: bool
    reasoning: str

# Guardrail agent to determine if the query is valid and classify it
guardrail_agent = Agent(
    name="Query Validator",
    instructions="Determine if the user query is appropriate to answer and classify it for routing.",
    output_type=TopicClassification,
)

# Guardrail function that runs before the main agent
async def topic_classifier_guardrail(ctx, agent, input_data):
    """Guardrail function to classify user queries and ensure they're appropriate."""
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(TopicClassification)
    
    # Not triggering the tripwire as we want the query to be processed
    # We're just using the guardrail to classify the topic
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=False,
    )

# ==========================================================================
# 4. DELEGATION WITH GUARDRAILS AND TRACKING
# ==========================================================================
# Enhanced delegation agent with guardrails and tracking
delegation_agent = Agent(
    name="Enhanced Delegation Agent",
    instructions="""You are the primary agent who receives user queries and determines which specialist to route them to.
For questions about documents, research papers, or information that might be in the knowledge base, delegate to the Document Knowledge Agent.
For questions requiring current information, news, or real-time data, delegate to the Web Search Agent.
Analyze the query carefully to make the appropriate routing decision.""",
    handoffs=[rag_agent, search_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=topic_classifier_guardrail),
    ],
)

# ==========================================================================
# 5. EXECUTION WITH TRACING
# ==========================================================================
async def main():
    """Run the demo with examples demonstrating guardrails and tracing."""
    # Configure tracing for the workflow
    run_config = RunConfig(
        workflow_name="Advanced Agent System with Guardrails",
        trace_include_sensitive_data=True
    )
    
    # Test with a document-related question
    document_query = "From the document you are provided with, tell me what is Brainli?"
    print("\nProcessing document query with guardrails and tracing...")
    
    result = await Runner.run(
        delegation_agent, 
        document_query,
        run_config=run_config
    )
    
    print(result)
    print("Document query trace ID:", run_config.trace_id)
    
    # Test with a web search question
    web_query = "What are the latest developments in AI in 2024?"
    print("\nProcessing web search query with guardrails and tracing...")
    
    # Create a new run config for distinct tracing
    web_run_config = RunConfig(
        workflow_name="Advanced Agent System with Guardrails",
        trace_include_sensitive_data=True
    )
    
    result = await Runner.run(
        delegation_agent, 
        web_query,
        run_config=web_run_config
    )
    
    print(result)
    print("Web query trace ID:", web_run_config.trace_id)

if __name__ == "__main__":
    asyncio.run(main()) 