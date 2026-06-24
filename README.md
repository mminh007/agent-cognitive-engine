# Agent Cognitive Engine

## Overview

Agent Cognitive Engine is an enterprise-grade multi-agent AI platform built on top of LangGraph, FastAPI, MCP (Model Context Protocol), ChromaDB, Redis, RabbitMQ, and OpenAI-compatible LLMs.

The platform provides intelligent task routing, domain-specific agent execution, tool orchestration, long-term memory management, semantic caching, hybrid retrieval (RAG), asynchronous memory extraction, and observability through Langfuse and LangSmith.

Designed for high-security enterprise integrations, the system supports both REST and gRPC interfaces, implements a zero-trust security architecture for background queues and MCP servers, and features asymmetric cryptographic receipts to guarantee AI response provenance for B2B partners.

---

## Key Features

### Multi-Agent Architecture
* Supervisor-based intent routing.
* Domain-specific agents (General Agent, Research Agent, Vision Agent).
* Dynamic agent selection using structured LLM outputs.

### LangGraph Orchestration
* Stateful workflow execution.
* Conditional routing & tool execution loops.
* **Conditional Session Persistence**:
  * **Authenticated Users**: Persists full chat history to Redis checkpointer and runs long-term fact extraction.
  * **Anonymous Users** (User ID prefix `anon_`): Runs purely in-memory without Redis checkpointer or background jobs, ensuring user privacy and database sanitization.

### Security & Zero-Trust Architecture
* **Claim Check Queue Pattern**:
  Fact extraction worker receives only reference payloads (`user_id`, `session_id`). It fetches the actual chat history directly from the trusted Redis checkpointer. This completely eliminates the risk of **Fact Poisoning** or **Long-term Memory Pollution** even if RabbitMQ is compromised.
* **MCP Handshake Verification (HS256 JWT)**:
  Secures local stdio MCP subprocesses using symmetric HS256 JWT tokens (`MCP_CLIENT_TOKEN` signed with `MCP_JWT_SECRET`). The MCP server verifies the token at boot, preventing unauthorized process execution.
* **AI Response Provenance (ECDSA ES256 Receipts)**:
  gRPC stream yields response text chunks and finishes with a cryptographic `Receipt` containing the accumulated response text hash and metadata, signed with our ECDSA private key. B2B partners verify the AI output's integrity, authenticity, and non-repudiation using only our Public Key.

### Hybrid RAG & Memory
* ChromaDB vector storage & OpenAI embeddings.
* BM25 lexical retrieval & Reciprocal Rank Fusion (RRF).
* Automated fact extraction, domain-isolated collections, and memory pruning.
* **Semantic Cache**: Similarity-based response caching to reduce LLM costs.

### Observability & Diagnostics
* Langfuse and LangSmith tracing.
* Prometheus metrics (active streams, error counters).
* Structured logging and diagnostics.

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Agent Framework** | LangGraph |
| **LLM Integration** | LangChain |
| **API Server** | FastAPI |
| **Streaming Engine**| gRPC Server Streaming + HTTP SSE |
| **Vector Database** | ChromaDB (Chroma Vector Store) |
| **Session Cache** | Redis (LangGraph checkpointer) |
| **Queue Broker** | RabbitMQ |
| **Tool Protocol** | MCP (Model Context Protocol) |
| **Observability** | Langfuse + LangSmith + Prometheus |
| **Security Cryptography**| ECDSA SECP256R1 (ES256) + HMAC HS256 |
| **Containerization**| Docker |

---

## High-Level Architecture

```text
                            ┌────────────────────┐
                            │     Client Apps    │
                            └─────────┬──────────┘
                                      │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
                  ▼                                         ▼
 
        ┌───────────────────┐                   ┌───────────────────┐
        │ FastAPI REST API  │                   │   gRPC Service    │
        └─────────┬─────────┘                   └─────────┬─────────┘
                  │                                       │
                  └─────────────────┬─────────────────────┘
                                    ▼
 
                     ┌─────────────────────────────┐
                     │      LangGraph Workflow     │
                     └──────────────┬──────────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
 
       ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
       │ General Agent  │ │ Research Agent │ │ Vision Agent   │
       └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
               │                  │                  │
               └──────────┬───────┴──────────┬───────┘
                          ▼                  ▼
 
                  ┌──────────────────────────────┐
                  │   MCP Tools (JWT Secured)    │
                  └──────────────┬───────────────┘
                                 │
                      ┌──────────┴──────────┐
                      ▼                     ▼
 
              ┌──────────────┐      ┌──────────────┐
              │ Search Tool  │      │ Custom Tools │
              └──────────────┘      └──────────────┘
 
                                 │
                                 ▼
 
                   ┌─────────────────────────┐
                   │ Hybrid Memory Retrieval │
                   └─────────────┬───────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
 
    ┌────────────┐      ┌────────────┐       ┌────────────┐
    │  ChromaDB  │      │   Redis    │       │  RabbitMQ  │
    │ Long-Term  │      │ Checkpoint │       │ Reference  │
    │   Memory   │      │ (Checkpt)  │       │ Payload Q  │
    └────────────┘      └────────────┘       └─────┬──────┘
                                                   │
                                                   ▼ (Loads history from DB)
                                             ┌────────────┐
                                             │   Worker   │
                                             └────────────┘
```

---

## Repository Structure

```text
agent/
├── app/
│   ├── api/             # REST controllers (chat, semantic cache)
│   ├── bootstrap/       # Startup hooks and dependency container
│   ├── core/            # Configuration settings, logger, metrics, JWT & ECDSA helpers
│   ├── graph/           # LangGraph workflow definitions, specialist nodes, state schema
│   ├── grpc_layer/      # Generated Protobuf Python files (chat_pb2, chat_pb2_grpc)
│   ├── mcp/             # MCP subprocess client manager, fastmcp server, custom tool domains
│   ├── retrieval/       # BM25 and Vector search engines for Hybrid RAG
│   └── services/        # RabbitMQ publisher, background workers, memory manager
│
├── protos/              # Protobuf API definitions (chat.proto)
├── docker-compose.yml
├── docker-compose.tracing.yml
├── docker-compose.studio.yml
├── dockerfile
├── requirements.txt
└── entrypoint.sh
```

---

## Running the Platform

### 1. Create Docker Network
```bash
docker network create agent_network
```

### 2. Configure Environment Variables
Create a `.env` file inside the `agent/` directory (refer to the **Environment Variables** section below for required keys, including `MCP_JWT_SECRET`, `SECURITY_AI_RECEIPT_PRIVATE_KEY`, etc.).

### 3. Start Langfuse Tracing Stack
```bash
cd ./agent
docker compose -f docker-compose.tracing.yml up --build -d
```
Langfuse UI will be available at: `http://localhost:3000` (Default setup).

### 4. Start Core Infrastructure & Microservices
```bash
cd ./agent
docker compose up --build -d
```
This boots:
* **Redis** (Short-term checkpoint manager)
* **RabbitMQ** (Secure reference-based task broker)
* **ChromaDB** (Long-term semantic store)
* **FastAPI Service** (SSE streaming REST endpoint)
* **gRPC Service** (Secure AI response streaming engine)
* **Memory Worker** (Background asynchronous fact extractor)
* **Prometheus & Grafana** (Monitoring dashboard on port `3001`)

---

## Environment Variables

Configure these variables inside your `agent/.env` file:

```env
# OpenAI / GitHub Models Configuration
OPENAI_API_KEY=github_...
OPENAI_API_BASE=https://models.inference.ai.azure.com
OPENAI_TIER1_FAST_MODEL=gpt-4o-mini
OPENAI_TIER2_BALANCED_MODEL=gpt-4o
OPENAI_TIER3_REASONING_MODEL=o1-mini

# MCP Subprocess Security
MCP_JWT_SECRET=your_secure_shared_hmac_secret_key_change_me

# B2B AI Response Provenance Signature (ECDSA SECP256R1 PEM keys)
SECURITY_AI_RECEIPT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..."
SECURITY_AI_RECEIPT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\n..."

# Redis Checkpointer
REDIS_URL=redis://redis_cache:6379/0
REDIS_TTL=3600

# ChromaDB Server Configuration
CHROMA_SERVER_HOST=chroma_server
CHROMA_SERVER_PORT=8000

# RabbitMQ Message Broker
RABBITMQ_URL=amqp://guest:guest@rabbitmq_broker:5672/
RABBITMQ_QUEUE_NAME=fact_extraction_queue

# Langfuse / Langsmith Tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://langfuse:3000
```

---

## MCP Extension & Secure Tool Load

MCP servers are registered in:
`app/mcp/mcp_servers_config.json`

Once configured, the `DynamicMcpClientManager` spawns the MCP process, signs a handshake JWT token, injects it into the environment as `MCP_CLIENT_TOKEN`, and the child server validates it before exporting any tools:

```json
[
  {
    "name": "internal-custom-tools",
    "command": "python",
    "args": ["-m", "app.mcp.server"]
  }
]
```

---

## Verification & Tests

To execute the cryptographic signature flow checks inside the Docker environment, run:
```bash
docker compose run --rm agent_grpc python test_grpc_receipt.py
```
This runs a simulated client stream and validates the ES256 receipt generation, text hashing, and public key verification.

---

## License

MIT License
