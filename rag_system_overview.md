# RAG System in ViktorAI

## What is RAG?
- **R**etrieval **A**ugmented **G**eneration
- Combines retrieval of relevant information with generative models
- Grounds LLM responses in factual data rather than just relying on model knowledge

## ViktorAI RAG Implementation

### Data Sources
- Character profile (viktor_profile.md)
- Character personality (viktor_personality.md)
- Character quotes (viktor_quotes.md)
- Character scenes and events (viktor_scenes_and_events.md)
- Character relationships (viktor_relationships.md)
- Character technology (viktor_technology.md)

### Vector Store
- Character data is processed into chunks with metadata
- Each chunk is converted to vector embeddings using Sentence Transformers
- FAISS library used for efficient vector search
- Fallback to simple vector store when FAISS not available

### Retrieval Process
1. User query is converted to a vector embedding
2. System finds most similar chunks in the vector store
3. Retrieves top k chunks (typically 3-5) most relevant to the query
4. Creates a context block from these chunks including their source

### Generation Process
1. Constructs a prompt using:
   - System prompt defining Viktor's character
   - Retrieved context from vector store
   - User's query
2. Sends complete prompt to LLM (e.g., gemma3:1b)
3. Generates response grounded in the retrieved context

## Key Benefits

### Character Consistency
- Responses are grounded in actual character information
- Reduces fabrication of events not in the show
- Maintains consistent character traits and history

### Accuracy
- References specific scenes and events from the show
- Uses actual quotes and characterization
- Provides technical details consistent with the Arcane universe

### Improved Responses
- More detailed and specific answers
- Better handling of obscure questions
- Reduced hallucination of non-canonical information

## Technical Implementation
- Built with sentence-transformers for embeddings
- Uses FAISS for vector search with numpy fallback
- Integrated with Ollama for LLM access
- Modular design allows swapping different components 