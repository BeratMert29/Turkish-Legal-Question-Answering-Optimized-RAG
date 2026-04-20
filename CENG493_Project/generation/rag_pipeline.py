import openai
import config

TURKISH_PROMPT = """Sen Türk hukuku alanında uzman bir hukuki asistansın. Görevin, yalnızca aşağıda numaralandırılmış [Kaynak N] bağlamlarını kullanarak soruyu eksiksiz ve doğru biçimde yanıtlamaktır.

ZORUNLU KURALLAR:
1. Yanıtını YALNIZCA verilen [Kaynak N] kaynaklarına dayandır. Kendi arka plan bilginden veya bağlamda yer almayan hiçbir bilgiden yararlanma.
2. İlgili her atıfta kanun adını VE madde numarasını açıkça belirt (örnek: "Türk Medeni Kanunu Madde 997", "Türk Ceza Kanunu Madde 53").
3. Birden fazla kaynak ilgiliyse hepsini sentezle ve [Kaynak N] numarasıyla göster.
4. Kaynaklar arasında çelişki varsa çelişkiyi açıkça ifade et ve her iki görüşü kaynak numarasıyla aktar.
5. Bağlam soruyu yanıtlamak için yetersizse "Sağlanan bağlam bu soruyu yanıtlamak için yeterli değildir." yaz; asla tahmin yürütme veya uydurma.

YANIT YAPISI:
- İlk cümle: Sorunun doğrudan yanıtı.
- Devamı: Hukuki dayanak — ilgili kanun adı, madde numarası ve bağlamdan alınan açıklama.
- Sonuç: Varsa pratik sonuç veya ek uyarı.

Yanıtını yalnızca Türkçe ver."""

SHORT_ANSWER_PROMPT = """Sen Türk hukuku alanında uzman bir hukuki asistansın. Sana bir soru ve bağlam verilecektir.

ZORUNLU KURALLAR:
1. Yanıtın yalnızca TEK bir ifade, sayı veya hukuki kavramdan oluşmalıdır — cümle kurma, açıklama yapma, gerekçe gösterme.
2. Yanıtı doğrudan bağlamdan çıkar; kendi bilgini kullanma.
3. Yanıt bağlamda yoksa yalnızca şunu yaz: Bilgi yok
4. Fazladan kelime, noktalama veya açıklama ekleme.

Yanıtını yalnızca Türkçe ver."""

class RAGPipeline:
    def __init__(self, retriever,
                 model: str = config.LLM_MODEL,
                 temperature: float = config.LLM_TEMPERATURE,
                 max_tokens: int = config.LLM_MAX_TOKENS,
                 top_k_for_generation: int = config.TOP_K_FOR_GENERATION,
                 context_window_chars: int = config.CONTEXT_WINDOW_CHARS,
                 short_answer_mode: bool = False):
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = SHORT_ANSWER_PROMPT if short_answer_mode else TURKISH_PROMPT
        self._effective_max_tokens = config.LLM_SHORT_ANSWER_MAX_TOKENS if short_answer_mode else max_tokens
        self.top_k_for_generation = top_k_for_generation
        self.context_window_chars = context_window_chars
        self._client = openai.OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )

    def get_llm_client(self) -> openai.OpenAI:
        """Return the shared OpenAI client pointing at Ollama endpoint."""
        return self._client

    def assemble_context(self, chunks: list) -> tuple[str, list]:
        """
        Format top_k_for_generation chunks as numbered sources.
        Returns (context_str, included_chunks) where included_chunks contains
        only the chunks whose text was not truncated out.
        """
        selected = chunks[:self.top_k_for_generation]
        parts = []
        included = []
        running_len = 0
        for i, chunk in enumerate(selected):
            part = f"[Kaynak {i+1}] ({chunk['source']})\n{chunk['text']}\n\n"
            if running_len + len(part) > self.context_window_chars:
                break
            parts.append(part)
            included.append(chunk)
            running_len += len(part)
        context = "".join(parts)
        return context, included

    def generate(self, question: str, context: str) -> str:
        """Generate answer via Ollama LLM."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self._effective_max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"},
            ],
        )
        if not response.choices:
            raise ValueError("LLM returned empty choices list")
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
        context_used, context_chunks = self.assemble_context(retrieved_chunks)
        answer = self.generate(question, context_used)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": context_chunks,   # only chunks that were in context
            "context_used": context_used,
        }
