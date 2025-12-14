"""Ingest markdown files into Qdrant vector database."""

import os
import re
import sys
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.embeddings import embed_texts
from src.config import get_settings

load_dotenv()
settings = get_settings()


def parse_markdown(content: str) -> Dict[str, Any]:
    """Parse markdown file and extract frontmatter and content.

    Args:
        content: Raw markdown file content

    Returns:
        Dict with title and content
    """
    # Extract frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    if frontmatter_match:
        content = content[frontmatter_match.end():]

    # Extract title from first H1
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled"

    return {
        "title": title,
        "content": content,
    }


def chunk_text(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk (approximated as words)
        overlap_tokens: Overlap between chunks

    Returns:
        List of chunk dicts with text and char positions
    """
    # Simple word-based chunking (approximation of tokens)
    words = text.split()
    chunks = []

    start_idx = 0
    char_pos = 0

    while start_idx < len(words):
        end_idx = min(start_idx + max_tokens, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        # Calculate character positions
        start_char = text.find(chunk_words[0], char_pos) if chunk_words else char_pos
        end_char = start_char + len(chunk_text)

        chunks.append({
            "text": chunk_text,
            "start_char": start_char,
            "end_char": end_char,
        })

        # Move to next chunk with overlap
        start_idx = end_idx - overlap_tokens
        char_pos = end_char

        if end_idx >= len(words):
            break

    return chunks


def extract_section_id(text: str, chapter_slug: str) -> str:
    """Extract section ID from text based on heading.

    Args:
        text: Chunk text
        chapter_slug: Parent chapter slug

    Returns:
        Section ID in format chapter_slug#heading_slug
    """
    # Find first heading in chunk
    heading_match = re.search(r'^##+ (.+)$', text, re.MULTILINE)
    if heading_match:
        heading = heading_match.group(1)
        heading_slug = re.sub(r'[^\w\s-]', '', heading.lower())
        heading_slug = re.sub(r'[\s_]+', '-', heading_slug)
        return f"{chapter_slug}#{heading_slug}"
    return chapter_slug


def ingest_docs(docs_path: str):
    """Ingest all markdown files from docs directory into Qdrant.

    Args:
        docs_path: Path to docs directory
    """
    docs_dir = Path(docs_path)

    if not docs_dir.exists():
        print(f"Error: Docs directory not found: {docs_dir}")
        sys.exit(1)

    # Find all markdown files
    md_files = sorted(docs_dir.glob("*.md"))

    if not md_files:
        print(f"Error: No markdown files found in {docs_dir}")
        sys.exit(1)

    print(f"Found {len(md_files)} markdown files")

    # Initialize Qdrant client
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    # Verify collection exists
    try:
        client.get_collection(settings.qdrant_collection)
        print(f"Using collection: {settings.qdrant_collection}")
    except Exception as e:
        print(f"Error: Collection '{settings.qdrant_collection}' not found.")
        print("Run setup_qdrant.py first to create the collection.")
        sys.exit(1)

    all_points = []
    total_chunks = 0

    for md_file in md_files:
        chapter_slug = md_file.stem  # filename without extension
        print(f"\nProcessing: {md_file.name}")

        # Read and parse file
        content = md_file.read_text(encoding="utf-8")
        parsed = parse_markdown(content)

        # Chunk the content
        chunks = chunk_text(parsed["content"])
        print(f"  Created {len(chunks)} chunks")

        # Prepare points for Qdrant
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "chapter_slug": chapter_slug,
                    "section_id": extract_section_id(chunk["text"], chapter_slug),
                    "chunk_text": chunk["text"],
                    "chunk_index": i,
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                },
            )
            all_points.append(point)

        total_chunks += len(chunks)

    # Upsert all points to Qdrant
    print(f"\nUploading {total_chunks} embeddings to Qdrant...")

    # Upload in batches of 100
    batch_size = 100
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i:i + batch_size]
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch,
        )
        print(f"  Uploaded batch {i // batch_size + 1}/{(len(all_points) + batch_size - 1) // batch_size}")

    print(f"\nIngestion complete!")
    print(f"  Total files: {len(md_files)}")
    print(f"  Total chunks: {total_chunks}")


def main():
    parser = argparse.ArgumentParser(description="Ingest textbook content into Qdrant")
    parser.add_argument(
        "--docs-path",
        type=str,
        default="../docs",
        help="Path to docs directory (default: ../docs)",
    )
    args = parser.parse_args()

    ingest_docs(args.docs_path)


if __name__ == "__main__":
    main()
