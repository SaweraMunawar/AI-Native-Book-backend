"""Generation service using Groq Llama 3."""

from typing import List, Optional

from groq import Groq

from src.config import get_settings
from src.services.retrieval import RetrievalResult

settings = get_settings()


SYSTEM_PROMPT = """You are an AI teaching assistant for the "Physical AI & Humanoid Robotics" textbook. Your role is to help students understand the textbook content.

CRITICAL RULES:
1. ONLY answer questions using the provided textbook context. Do not use any external knowledge.
2. If the context does not contain enough information to answer the question, say "This topic may not be covered in this textbook" or "I don't have enough information from the textbook to answer this."
3. Always cite the specific chapter when providing information.
4. Keep answers clear, concise, and educational.
5. If asked about topics outside the textbook scope, politely redirect to the textbook content.

The textbook covers:
- Chapter 1: Introduction to Physical AI
- Chapter 2: Basics of Humanoid Robotics
- Chapter 3: ROS 2 Fundamentals
- Chapter 4: Digital Twin Simulation (Gazebo + Isaac Sim)
- Chapter 5: Vision-Language-Action Systems
- Chapter 6: Capstone Project"""


def get_groq_client() -> Groq:
    """Get Groq client instance."""
    return Groq(api_key=settings.groq_api_key)


def format_context(results: List[RetrievalResult]) -> str:
    """Format retrieval results as context for the LLM.

    Args:
        results: List of retrieval results

    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant content found in the textbook."

    context_parts = []
    for i, result in enumerate(results, 1):
        chapter_name = result.chapter_slug.replace("-", " ").title()
        context_parts.append(
            f"[Source {i}: {chapter_name}]\n{result.chunk_text}"
        )

    return "\n\n".join(context_parts)


def generate_response(
    query: str,
    context_results: List[RetrievalResult],
    selected_text: Optional[str] = None,
) -> str:
    """Generate response using Groq Llama 3.

    Args:
        query: User's question
        context_results: Retrieved context from vector search
        selected_text: Optional selected text for contextual queries

    Returns:
        Generated response text
    """
    client = get_groq_client()

    # Build context
    context = format_context(context_results)

    # Build user message
    if selected_text:
        user_message = f"""The student has selected the following text from the textbook:

"{selected_text}"

They are asking: {query}

Relevant textbook context:
{context}

Please help the student understand the selected text and answer their question based on the textbook content."""
    else:
        user_message = f"""Student question: {query}

Relevant textbook context:
{context}

Please answer the student's question based on the textbook content provided."""

    # Generate response
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=settings.groq_model,
        temperature=0.3,
        max_tokens=1000,
    )

    return chat_completion.choices[0].message.content


def generate_low_confidence_response(query: str) -> str:
    """Generate response for low confidence queries.

    Args:
        query: User's original question

    Returns:
        Helpful response indicating topic may not be covered
    """
    return (
        "I couldn't find enough relevant information in the textbook to answer "
        "your question about this topic. This may be because:\n\n"
        "1. The topic isn't covered in this textbook\n"
        "2. The question is phrased differently than the textbook content\n"
        "3. This is an advanced topic beyond the scope of this course\n\n"
        "Try rephrasing your question, or ask about specific topics from the "
        "table of contents: Introduction to Physical AI, Humanoid Robotics, "
        "ROS 2, Digital Twins, or VLA Systems."
    )
