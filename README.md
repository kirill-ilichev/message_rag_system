# Message RAG System

A Retrieval-Augmented Generation system that processes messages and answers user questions with source links using semantic search.

## Features

- 🔍 **Semantic Search**: Uses OpenAI embeddings and FAISS for efficient similarity search
- 📝 **Structured Output**: Generates answers with guaranteed source citations using Pydantic models
- 🎯 **Author Attribution**: Includes message authors in source descriptions
- ⚡ **Fast & Scalable**: FAISS vector database for high-performance retrieval
- 🔒 **Secure**: API keys managed via environment variables

## Technology Stack

- **Language**: Python 3.13+
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o-mini with structured outputs

## Installation

### Prerequisites

- Python 3.13
- Poetry (recommended) or pip
- OpenAI API key

### Setup

1. **Clone the repository**

2. **Install dependencies**

Using Poetry (recommended):
```bash
poetry install
poetry shell
```

Using pip:
```bash
pip install numpy pydantic faiss-cpu openai
```

3. **Set up your OpenAI API key**

On macOS/Linux:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

On Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your-api-key-here
```

On Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

For permanent setup, add the key to your `.bashrc`, `.zshrc`, or system environment variables.

## Usage

### 1. Build the Index

First, build the vector index from your messages JSON file:

```bash
python main.py --build messages.json
```

This will:
- Read all messages from `messages.json`
- Generate embeddings using OpenAI
- Create a FAISS index
- Save artifacts to `artifacts_small/` directory

**Custom output directory:**
```bash
python main.py --build messages.json --artifacts my_custom_path
```

### 2. Ask Questions

Query the system with natural language questions:

```bash
python main.py --ask "What was discussed about the budget and timeline?"
```

**Example output:**
```
The team discussed allocating $50k for development and $20k for marketing. 

The project timeline is set for target delivery in Q2 2025.

Sources:

- Budget allocation discussion by John (https://example.com/msg_042)

- Timeline proposal by Sarah (https://example.com/msg_045)
```

### 3. Use Custom Artifacts Directory

```bash
# Build with custom directory
python main.py --build messages.json --artifacts ./my_index

# Query using custom directory
python main.py --ask "What did Sarah say?" --artifacts ./my_index
```

## Input Format

Messages should be in JSON format with the following structure:

```json
[
  {
    "message_id": "msg_001",
    "url": "https://example.com/messages/msg_001",
    "author": "John Doe",
    "timestamp": "2025-01-15T10:30:00Z",
    "content": "Message text here...",
    "metadata": {
      "channel": "general",
      "tags": ["important", "announcement"]
    }
  }
]
```

### Required Fields

- `message_id`: Unique identifier for the message
- `url`: Link to the original message
- `author`: Message author name
- `content`: The actual message text

### Optional Fields

- `timestamp`: When the message was sent
- `metadata`: Additional information (channel, tags, etc.)

## Example Queries

Here are some example questions you can ask:

### Specific Facts
```bash
python main.py --ask "What is the budget allocation for development?"
python main.py --ask "When is the project timeline set for?"
python main.py --ask "Who discussed the marketing budget?"
```

### Topic-Based
```bash
python main.py --ask "What security updates were mentioned?"
python main.py --ask "Tell me about performance improvements"
python main.py --ask "What training sessions are available?"
```

### Person-Specific
```bash
python main.py --ask "What did Sarah announce?"
python main.py --ask "What did John say about the budget?"
```

### Time-Based
```bash
python main.py --ask "What happened in January 2025?"
python main.py --ask "What Q1 achievements were mentioned?"
```

## How It Works

1. **Indexing Phase** (`--build`):
   - Reads messages from JSON file
   - Extracts content, URLs, authors, and IDs
   - Generates embeddings using OpenAI's embedding model
   - Stores vectors in FAISS index
   - Saves metadata for retrieval

2. **Query Phase** (`--ask`):
   - Converts user question to embedding
   - Searches FAISS index for top-k similar messages
   - **Filters results by similarity threshold** (removes irrelevant results)
   - Retrieves relevant messages with metadata
   - Generates structured answer using GPT-4o-mini
   - Formats output with sources and authors

## Relevance Filtering

The system uses a **similarity threshold** to filter out irrelevant results and avoid misleading responses.

### How It Works

- Each search result gets a **similarity score** (0-1) based on cosine similarity
- Only results above the threshold are included in the answer
- If no results pass the threshold, you get a clear message instead of irrelevant sources

### Example: Irrelevant Query

```bash
python main.py --ask "What do you know about dinosaurs?"
```

**Output:**
```
No relevant messages found for: 'What do you know about dinosaurs?'

This question appears to be unrelated to the available messages.
Try asking about topics covered in your message database.
```

This prevents the system from showing random, unrelated sources just to provide an answer.

## Configuration

### Default Settings

- **Embedding Model**: `text-embedding-3-small`
- **LLM Model**: `gpt-4o-mini`
- **Retrieval Count**: Top 5 most relevant messages
- **Similarity Threshold**: 0.3 (on a scale of 0-1)
- **Temperature**: 0.2 (for consistent answers)
- **Max Tokens**: 400

### Advanced Options

#### Adjusting the Similarity Threshold

Control how strict the relevance filtering is:

```bash
# Stricter (only very relevant results)
python main.py --ask "your question" --threshold 0.5

# Default (balanced)
python main.py --ask "your question" --threshold 0.3

# More permissive (more results, potentially less relevant)
python main.py --ask "your question" --threshold 0.2
```

**Guidelines:**
- **0.5-0.7**: High precision, strict filtering (best for critical queries)
- **0.3-0.5**: Balanced (recommended for general use)
- **0.2-0.3**: High recall, permissive (when you don't want to miss anything)

#### Debug Mode: Show Similarity Scores

See the similarity scores for retrieved messages to understand relevance:

```bash
python main.py --ask "What was the budget?" --show-scores
```

**Output:**
```
Retrieved messages (with similarity scores):
  - Score: 0.8523 | John: We discussed the budget allocation and agreed on $50k for development...
  - Score: 0.7234 | Sarah: Confirming the project timeline: target delivery is set for Q2 2025...
  - Score: 0.4521 | Carol: Marketing campaign draft is ready for feedback. Check your email...

The team discussed allocating $50k for development and $20k for marketing. 

The project timeline is set for target delivery in Q2 2025.

Sources:

- Budget allocation discussion by John (https://example.com/msg_042)
- Timeline proposal by Sarah (https://example.com/msg_045)
```

This helps you:
- Understand why certain results were included
- Tune the threshold for your use case
- Debug unexpected results
