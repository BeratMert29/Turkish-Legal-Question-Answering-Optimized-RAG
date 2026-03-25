import openai
import config

TURKISH_PROMPT = """Sen bir Türk hukuk uzmanısın. Sana verilen bağlam bilgilerini kullanarak
soruyu doğru, eksiksiz ve kaynaklara dayalı olarak yanıtla.
Eğer bağlam bilgisi yetersizse, bunu açıkça belirt.
Yanıtını Türkçe ver."""

class RAGPipeline:
    def __init__(self, retriever,
                 model: str = config.LLM_MODEL,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS,
                 top_k_for_generation: int = config.TOP_K_FOR_GENERATION,
                 context_window_chars: int = config.CONTEXT_WINDOW_CHARS):
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k_for_generation = top_k_for_generation
        self.context_window_chars = context_window_chars
        self._client = openai.OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )

    def get_llm_client(self) -> openai.OpenAI:
        """Return the shared OpenAI client pointing at Ollama endpoint."""
        return self._client

    def assemble_context(self, chunks: list) -> str:
        """
        Format top_k_for_generation chunks as numbered sources.
        Each chunk: "[Kaynak {i+1}] ({source})\n{text}\n\n"
        Concatenate then truncate to context_window_chars.
        chunks is list of RetrievedChunk TypedDicts — use dict access: chunk["source"], chunk["text"]
        """
        selected = chunks[:self.top_k_for_generation]
        parts = [
            f"[Kaynak {i+1}] ({chunk['source']})\n{chunk['text']}\n\n"
            for i, chunk in enumerate(selected)
        ]
        context = "".join(parts)
        return context[:self.context_window_chars]

    def generate(self, question: str, context: str) -> str:
        """Generate answer via Ollama LLM."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": TURKISH_PROMPT},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response")
        return content.strip()

    def run(self, question: str, top_k_retrieval: int = config.TOP_K_RETRIEVAL) -> dict:
        """
        Full RAG pipeline: retrieve → assemble context → generate.

        Returns:
            {
                "question": str,
                "answer": str,
                "retrieved_chunks": list[RetrievedChunk],   # ← key is "retrieved_chunks"
                "context_used": str,
            }
        """
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k_retrieval)
        context_used = self.assemble_context(retrieved_chunks)
        answer = self.generate(question, context_used)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "context_used": context_used,
        }
