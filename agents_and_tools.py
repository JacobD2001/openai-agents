import asyncio
from agents import (
    Agent, Runner, FileSearchTool, WebSearchTool, RunConfig
)

# ==========================================================================
# AGENTS AND TOOLS TUTORIAL
# ==========================================================================
# This tutorial demonstrates the basic concepts of:
# 1. Creating tools for agents
# 2. Defining specialist agents with specific capabilities
# 3. Setting up agent handoffs for delegation

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
# 3. DELEGATION WITH HANDOFFS
# ==========================================================================
# Delegation agent that routes queries to appropriate specialist agents
delegation_agent = Agent(
    name="Delegation Agent",
    instructions="""You are the primary agent who receives user queries and determines which specialist to route them to.
For questions about documents, research papers, or information that might be in the knowledge base, delegate to the Document Knowledge Agent.
For questions requiring current information, news, or real-time data, delegate to the Web Search Agent.
Analyze the query carefully to make the appropriate routing decision.""",
    handoffs=[rag_agent, search_agent],
)

# ==========================================================================
# DEMO EXECUTION
# ==========================================================================
async def main():
    """Run the demo with examples of document-based and web search queries."""
    # Test with a document-related question
    document_query = "From the document you are provided with, tell me what is Brainli?"
    
    result = await Runner.run(
        delegation_agent, 
        document_query
    )
    
    print(result)
    
    # Test with a web search question
    web_query = "What are the latest developments in AI in 2024?"
    
    result = await Runner.run(
        delegation_agent, 
        web_query
    )
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())