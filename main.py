import asyncio
from pydantic import BaseModel
from agents import (
    Agent, InputGuardrail, GuardrailFunctionOutput, Runner, 
    FileSearchTool, WebSearchTool, function_tool, RunConfig
)

# ==========================================================================
# 1. DEFINING AGENTS AND TOOLS
# ==========================================================================
# Documents must be prepared for RAG using the vectorize_docs.py script, which:
# - Uploads a document to OpenAI (e.g., test.pdf)
# - Creates a vector store
# - Adds the document to the vector store
# - Returns a vector store ID to use in the FileSearchTool

# Create a FileSearchTool for document-based knowledge agent
document_file_search = FileSearchTool(
    vector_store_ids=["vs_67d8a6a7c9c4819187ffcdd44d1cd330"],  # Replace with your vector store ID
    max_num_results=3,
)

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

# Create a WebSearchTool for the web search agent
web_search = WebSearchTool()

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

# Delegation agent that routes queries to appropriate specialist agents
delegation_agent = Agent(
    name="Delegation Agent",
    instructions="""You are the primary agent who receives user queries and determines which specialist to route them to.
For questions about documents, research papers, or information that might be in the knowledge base, delegate to the Document Knowledge Agent.
For questions requiring current information, news, or real-time data, delegate to the Web Search Agent.
Analyze the query carefully to make the appropriate routing decision.""",
    handoffs=[rag_agent, search_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=topic_classifier_guardrail),  # Defined below
    ],
)

# ==========================================================================
# 2. IMPLEMENTING GUARDRAILS
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
# MAIN FUNCTION - RUNNING THE DEMO
# ==========================================================================
async def main():
    """Run the demo with examples of document-based and web search queries."""
    # Test with a document-related question
    document_query = "From the document you are provided with, tell me what is the deep research capability?"
    
    result = await Runner.run(
        delegation_agent, 
        document_query,
        run_config=RunConfig(
            workflow_name="Document and Web Information Assistant",
            trace_include_sensitive_data=True
        )
    )
    
    # Test with a web search question (current information)
    web_query = "What are the latest developments in AI in 2024?"
    
    result = await Runner.run(
        delegation_agent, 
        web_query,
        run_config=RunConfig(
            workflow_name="Document and Web Information Assistant",
            trace_include_sensitive_data=True
        )
    )

if __name__ == "__main__":
    asyncio.run(main())