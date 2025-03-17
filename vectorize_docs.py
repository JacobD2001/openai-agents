import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def vectorize_document():
    """
    Main function to upload and vectorize a document for use with FileSearchTool.
    Returns the vector store ID upon successful completion.
    """
    # Initialize OpenAI client
    client = OpenAI()
    
    # Validate API key is set
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(1)
    
    # -------------------------------------------------------------
    # Step 1: Upload Document to OpenAI
    # -------------------------------------------------------------
    try:
        with open("test.pdf", "rb") as file_content:
            file_upload = client.files.create(
                file=file_content,
                purpose="assistants"
            )
        file_id = file_upload.id
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)
    
    # -------------------------------------------------------------
    # Step 2: Create Vector Store
    # -------------------------------------------------------------
    try:
        vector_store = client.vector_stores.create(
            name="document_knowledge_base"
        )
        vector_store_id = vector_store.id
    except Exception as e:
        sys.exit(1)
    
    # -------------------------------------------------------------
    # Step 3: Add Document to Vector Store
    # -------------------------------------------------------------
    try:
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_id
        )
    except Exception as e:
        sys.exit(1)
    
    # -------------------------------------------------------------
    # Step 4: Check Processing Status
    # -------------------------------------------------------------
    try:
        client.vector_stores.files.list(
            vector_store_id=vector_store_id
        )
    except Exception:
        pass
    
    return vector_store_id

def check_status(vector_store_id):
    """
    Check the processing status of files in a vector store.
    Use this to verify when documents are ready for retrieval.
    
    Args:
        vector_store_id: The ID of the vector store to check
    """
    client = OpenAI()
    try:
        file_status = client.vector_stores.files.list(
            vector_store_id=vector_store_id
        )
        
        if not file_status.data:
            return
    except Exception:
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-only":
        if len(sys.argv) > 2:
            check_status(sys.argv[2])
        else:
            sys.exit(1)
    else:
        vectorize_document() 