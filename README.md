# Agent Cognitive Orchestrator

## Overview

Agent Cognitive Orchestrator is an enterprise-grade multi-agent AI platform built on top of LangGraph, FastAPI, MCP (Model Context Protocol), ChromaDB, Redis, RabbitMQ, and OpenAI-compatible LLMs.

The platform provides intelligent task routing, domain-specific agent execution, tool orchestration, long-term memory management, semantic caching, hybrid retrieval (RAG), asynchronous memory extraction, and observability through Langfuse and LangSmith.

Designed for scalability and extensibility, the system supports both REST and gRPC interfaces while allowing new capabilities to be added dynamically through MCP servers.

---

## Key Features

### Multi-Agent Architecture

* Supervisor-based intent routing
* Domain-specific agents:

  * General Agent
  * Research Agent
  * Vision Agent
* Dynamic agent selection using structured LLM outputs

### LangGraph Orchestration

* Stateful workflow execution
* Conditional routing
* Tool execution loops
* Redis checkpoint persistence

### MCP (Model Context Protocol)

* Dynamic tool loading
* External tool integration
* Runtime tool discovery
* Independent MCP server deployment

### Hybrid RAG System

* ChromaDB vector storage
* OpenAI embeddings
* BM25 lexical retrieval
* Reciprocal Rank Fusion (RRF)

### Long-Term Memory

* Automated fact extraction
* Domain-isolated memory collections
* Semantic memory retrieval
* Memory pruning policies

### Semantic Cache

* Similarity-based response caching
* Reduced LLM cost
* Faster response generation

### Async Event Processing

* RabbitMQ message broker
* Background memory extraction
* Dead Letter Queue (DLQ) support

### API Layer

* FastAPI REST endpoints
* Server-Sent Events (SSE) streaming
* gRPC streaming service

### Observability

* Langfuse tracing
* LangSmith tracing
* Structured logging
* Runtime diagnostics

### Computer Vision Ready

* Image ingestion endpoints
* Object detection pipeline foundation
* Future multimodal expansion

---

## Technology Stack

| Layer            | Technology           |
| ---------------- | -------------------- |
| Agent Framework  | LangGraph            |
| LLM Integration  | LangChain            |
| API              | FastAPI              |
| Streaming        | SSE + gRPC           |
| Memory Store     | ChromaDB             |
| Session Store    | Redis                |
| Queue System     | RabbitMQ             |
| Tool Protocol    | MCP                  |
| Embeddings       | OpenAI Embeddings    |
| Tracing          | Langfuse + LangSmith |
| Containerization | Docker               |
| Scheduler        | APScheduler          |

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
                 │         MCP Tools            │
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
   │  ChromaDB  │      │   Redis    │       │ RabbitMQ   │
   │ Long-Term  │      │ Checkpoint │       │ Async Jobs │
   └────────────┘      └────────────┘       └────────────┘

                                │
                                ▼

                 ┌──────────────────────────┐
                 │ Langfuse / LangSmith     │
                 │ Observability & Tracing  │
                 └──────────────────────────┘
```

---

## Repository Structure

```text
agent/
│
├── app/
│   ├── api/
│   ├── graph/
│   ├── services/
│   ├── mcp/
│   ├── core/
│   └── grpc_server.py
│
├── docker-compose.yml
├── docker-compose.tracing.yml
├── docker-compose.studio.yml
│
├── dockerfile
├── dockerfile.studio
│
├── requirements.txt
├── entrypoint.sh
└── README.md
```

---

## Running the Platform

### 1. Create Docker Network

```cmd
docker network create agent_network
```

---

### 2. Start Langfuse Tracing Stack

```cmd
cd ./agent

docker compose -f docker-compose.tracing.yml up --build -d
```

This will start:

* PostgreSQL
* Langfuse Server

Langfuse UI:

```text
http://localhost:3000
```

---

### 3. Start Core Infrastructure

```cmd
cd ./agent

docker compose up --build -d
```

This will start:

* Redis
* RabbitMQ
* ChromaDB
* FastAPI Service
* gRPC Service
* Memory Worker

---

### 4. Start LangGraph Studio (Optional)

```cmd
cd ./agent

docker compose -f docker-compose.studio.yml up --build -d
```

LangGraph Studio:

```text
http://localhost:8123
```

---

## Stopping Services

### Stop Core Platform

```cmd
docker compose down
```

### Stop Tracing Stack

```cmd
docker compose -f docker-compose.tracing.yml down
```

### Stop LangGraph Studio

```cmd
docker compose -f docker-compose.studio.yml down
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI / GitHub Models
OPENAI_API_KEY=github_...
OPENAI_API_BASE=https://models.inference.ai.azure.com
OPENAI_DEFAULT_MODEL=gpt-4o-mini
OPENAI_MAX_COMPLETION_TOKENS=4096
OPENAI_MAX_CONTEXT_TOKENS=8192
OPENAI_TEMPERATURE=0.7

# Tavily Search
TAVILY_API_KEY=tvly-...
TAVILY_MAX_TOKEN_BUDGET=2000

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_TTL=3600

# ChromaDB
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION_NAME=long_term_memory

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672/
RABBITMQ_QUEUE_NAME=fact_extraction_queue

# Logging
LOGS_DIR=./logs
LOGS_MAX_BYTES=10485760
LOGS_BACKUP_COUNT=5

# LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=http://localhost:1984
LANGCHAIN_API_KEY=foo
LANGCHAIN_PROJECT=agent-ecosystem-prod

# Langfuse Tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_REPOSITORIES_TELEMETRY_ENABLED=false
```

---

## MCP Extension Example

Register a new MCP server inside:

```text
app/mcp/mcp_servers_config.json
```

Example:

```json
[
  {
    "name": "internal-custom-tools",
    "command": "python",
    "args": ["-m", "app.mcp.server"]
  }
]
```

Once the server is started, all exported MCP tools are automatically discovered and injected into the LangGraph workflow.

---

## Future Roadmap

* Multi-modal Vision Agent
* Knowledge Graph Memory Layer
* Multi-tenant Authentication
* Distributed Agent Clusters
* Kubernetes Deployment
* Advanced MCP Marketplace
* Autonomous Planning Agents
* Human-in-the-Loop Workflows

---

## License

MIT License

---

## Author

Agent Cognitive Orchestrator Team
