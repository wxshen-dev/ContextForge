# ContextForge

ContextForge imports PDF and Markdown documents into a searchable knowledge base and answers questions through a browser chat interface. The import and query services run separately so document processing does not block chat requests.

## What it does

- Imports PDF or Markdown files, extracts text and images, splits documents into chunks, creates BGE-M3 embeddings, and writes chunks to Milvus.
- Identifies product names from documents and uses them to narrow retrieval.
- Answers questions using conversation history, vector retrieval, optional web search, reranking, and a language model.
- Stores chat history in MongoDB and can stream answers and progress to the browser.

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- Docker Desktop for the included local MongoDB, Milvus, and MinIO services.
- An OpenAI-compatible model endpoint and a local BGE-M3 model for document import and answers. A local reranker model is recommended; the query workflow falls back to its initial ranking if the reranker cannot load.
- MinerU for PDF conversion. The web-search MCP endpoint is optional.

Install dependencies:

```powershell
uv sync --locked
Copy-Item .env.example .env
```

Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_DEFAULT_MODEL`, and `BGE_M3_PATH` in `.env`. Set `BGE_DEVICE` to `cpu` or a working CUDA device. `LLM_DEFAULT_TEMPERATURE` must contain a number. The `.env` file is ignored by Git. For PDF files, also configure MinerU. The web-search MCP endpoint is optional.

Start the local storage services from the repository root:

```powershell
docker compose -f compose.local.yml up -d mongo minio milvus
docker compose -f compose.local.yml ps
```

The compose file binds MongoDB to `127.0.0.1:27017`, the application MinIO API to `127.0.0.1:39000` (console: `39001`), and Milvus to `127.0.0.1:39530`. These match `.env.example`. MinIO uses the credentials in `.env`; change the example values before using this setup outside local development. Docker volumes retain data between restarts. Stop the stack with `docker compose -f compose.local.yml down`.

## Run locally

Start the two services in separate terminals from the repository root:

```powershell
uv run contextforge import
uv run contextforge query
```

The import page is at `http://127.0.0.1:8000/import.html`; the chat page is at `http://127.0.0.1:8001/chat.html`. Each page links to the other. Use `--host` or `--port` to override a default. Each service also exposes `/docs` for its API schema.

## API

| Service | Endpoint | Purpose |
| --- | --- | --- |
| Import | `POST /upload` | Upload one or more PDF/Markdown files as multipart `files`. |
| Import | `GET /status/{task_id}` | Read import progress and status. |
| Query | `GET /health` | Check that the HTTP service is running. |
| Query | `POST /query` | Ask a question with `query`, optional `session_id`, and `is_stream`. |
| Query | `GET /stream/{session_id}` | Receive progress and answer events when streaming is enabled. |
| Query | `GET /status/{session_id}` | Read query progress and the latest answer. |
| Query | `GET /history/{session_id}` | Read recent conversation messages. |
| Query | `DELETE /history/{session_id}` | Clear a conversation. |

Example synchronous request:

```powershell
$body = @{ query = 'How do I measure voltage?'; is_stream = $false } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8001/query -Method Post -ContentType 'application/json' -Body $body
```

## Browser acceptance

The included [sample document](tests/fixtures/atlas-m42.md) makes the workflow reproducible. With the local Docker stack and model settings above, upload it from the import page and wait for **Completed**. The activity log should show upload, Markdown processing, splitting, product identification, vector generation, and Milvus import. In the chat page, ask: **How do I measure DC voltage with the Atlas M42?** A successful answer includes the COM and VΩ jacks, DC V setting, probes in parallel, and 600 V maximum range. Refresh the page to verify that MongoDB restores the conversation.

This workflow was exercised in a real browser with the running services. The sample import completed, product and document vectors were stored in Milvus, a live query returned a document-based answer, and chat history was read back from MongoDB. The screenshots below capture the live browser states.

![Completed document import](docs/screenshots/import-accepted.png)

![Document question and answer](docs/screenshots/chat-accepted.png)

PDF conversion additionally requires MinerU. Knowledge-graph querying is not wired into the current answer workflow.
