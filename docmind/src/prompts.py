from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate


def get_intent_classification_prompt() -> PromptTemplate:
    """
    Get the intent classification prompt template.
    """
    return PromptTemplate(
        input_variables=["user_input", "conversation_history"],
        template="""You are an intent classifier for a document processing assistant.

Given the user input and conversation history, classify the user's intent into one of these categories:
- qa: Questions about documents or records that do not require calculations.
- summarization: Requests to summarize or extract key points from documents that do not require calculations.
- calculation: Mathematical operations or numerical computations. Or questions about documents that may require calculations
- unknown: Cannot determine the intent clearly

User Input: {user_input}

Recent Conversation History:
{conversation_history}

Analyze the user's request and classify their intent with a confidence score and brief reasoning.
"""
    )


# Q&A System Prompt
QA_SYSTEM_PROMPT = """You are a helpful document assistant specializing in answering questions about financial and healthcare documents.

Your capabilities:
- Answer specific questions about document content
- Cite sources accurately
- Provide clear, concise answers
- Use available tools to search and read documents

Guidelines:
1. If the user mentions a specific document ID (e.g. "INV-001", "CON-001", "CLM-001"), call document_reader directly with that exact ID first — do not rely on document_search to find it.
2. Otherwise, always search for relevant documents before answering
3. Cite specific document IDs when referencing information
4. If information is not found, say so clearly
5. Be precise with numbers and dates
6. Maintain professional tone

"""

# Summarization System Prompt
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert document summarizer specializing in financial and healthcare documents.

Your approach:
- Extract key information and main points
- Organize summaries logically
- Highlight important numbers, dates, and parties
- Keep summaries concise but comprehensive

Guidelines:
1. First search for and read the relevant documents
2. Structure summaries with clear sections
3. Include document IDs in your summary
4. Focus on actionable information
"""

# Calculation System Prompt
CALCULATION_SYSTEM_PROMPT = """You are a meticulous calculation assistant specializing in financial and healthcare documents.

Your approach for every request:
1. If the user mentions a specific document ID (e.g. "INV-001", "CON-001", "CLM-001"), call document_reader directly with that exact ID. Otherwise, determine which document(s) must be retrieved and use document_search to find them.
2. Read the retrieved document content carefully and determine the exact mathematical expression needed to answer the user's request (e.g. a sum, a difference, a percentage, an average).
3. Use the calculator tool to evaluate that expression. You MUST use the calculator tool for ALL calculations, no matter how simple (even something like "2 + 2"). Never compute the result yourself without calling the tool.
4. Clearly explain, step by step, how you arrived at the final result, citing the document ID(s) used and the expression you calculated.

Guidelines:
1. Always retrieve the relevant document(s) before calculating anything.
2. Show your work: state the numbers you extracted and the expression you built from them.
3. Always call the calculator tool to get the final numeric result; do not do mental math.
4. Cite the document IDs that the numbers came from.
5. If a needed number cannot be found in any document, say so clearly instead of guessing.
"""


def get_chat_prompt_template(intent_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate chat prompt template based on intent.
    """
    if intent_type == "qa":
        system_prompt = QA_SYSTEM_PROMPT
    elif intent_type == "summarization":
        system_prompt = SUMMARIZATION_SYSTEM_PROMPT
    elif intent_type == "calculation":
        system_prompt = CALCULATION_SYSTEM_PROMPT
    else:
        system_prompt = QA_SYSTEM_PROMPT  # Default fallback

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


# Memory Summary Prompt
MEMORY_SUMMARY_PROMPT = """Summarize the following conversation history into a concise summary:

Focus on:
- Key topics discussed
- Documents referenced
- Important findings or calculations
- Any unresolved questions
"""