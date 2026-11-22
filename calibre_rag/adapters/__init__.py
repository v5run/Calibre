"""Integrations (planned, v1+): LangChain retriever, LlamaIndex postprocessor,
Weaviate / Qdrant client-side fusion helpers, smolagents / MCP search tool (v4).

Every adapter does the same thing: fetch the BM25 and dense result lists with
scores, convert them to `calibre_rag.Hit`, and call `Calibre.retrieve`.
"""
