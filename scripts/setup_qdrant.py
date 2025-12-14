"""Setup Qdrant Cloud collection for textbook embeddings."""

import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

load_dotenv()


def setup_qdrant_collection():
    """Create the textbook_embeddings collection in Qdrant Cloud."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be set in .env")
        sys.exit(1)

    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    collection_name = "textbook_embeddings"

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        print(f"Collection '{collection_name}' already exists.")
        response = input("Do you want to delete and recreate it? (yes/no): ")
        if response.lower() == "yes":
            client.delete_collection(collection_name)
            print(f"Deleted collection '{collection_name}'.")
        else:
            print("Keeping existing collection.")
            return

    # Create collection with 384-dim vectors (MiniLM-L6-v2)
    print(f"Creating collection '{collection_name}' with 384-dim vectors...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    # Create payload index for filtered search
    print("Creating payload index for chapter_slug...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chapter_slug",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    print(f"Successfully created collection '{collection_name}'!")
    print("\nPayload schema:")
    print("  - chapter_slug: string (indexed)")
    print("  - section_id: string")
    print("  - chunk_text: string")
    print("  - chunk_index: integer")
    print("  - start_char: integer")
    print("  - end_char: integer")


if __name__ == "__main__":
    setup_qdrant_collection()
