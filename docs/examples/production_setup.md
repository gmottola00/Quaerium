# Production Setup Guide

Deploy your RAG application to production with Docker, monitoring, scaling, and best practices.

## Docker Deployment

### Complete Docker Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Milvus vector database
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

  # RAG API service
  rag-api:
    build: .
    container_name: rag-api
    ports:
      - "8000:8000"
    environment:
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - milvus
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: rag-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: rag-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: rag-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  etcd_data:
  minio_data:
  milvus_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Requirements

```txt
# requirements.txt
rag-toolkit>=0.1.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
redis>=5.0.0
prometheus-client>=0.19.0
python-multipart>=0.0.6
aiofiles>=23.2.1
```

## Production Application

### FastAPI Service

```python
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis.asyncio as redis
import hashlib
import json
from rag_toolkit import RagPipeline
from rag_toolkit.infra.embedding import OpenAIEmbedding
from rag_toolkit.infra.llm import OpenAILLM
from rag_toolkit.core.vectorstore import MilvusVectorStore
from prometheus_client import Counter, Histogram, generate_latest
import time
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
QUERY_COUNTER = Counter('rag_queries_total', 'Total RAG queries')
QUERY_LATENCY = Histogram('rag_query_latency_seconds', 'Query latency')
CACHE_HITS = Counter('rag_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('rag_cache_misses_total', 'Cache misses')

app = FastAPI(title="RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis cache
cache: redis.Redis = None

# RAG pipeline
pipeline: RagPipeline = None

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    use_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    cached: bool = False
    latency_ms: float

@app.on_event("startup")
async def startup():
    """Initialize services."""
    global cache, pipeline
    
    # Redis cache
    cache = await redis.from_url(
        "redis://redis:6379",
        encoding="utf-8",
        decode_responses=True
    )
    
    # RAG pipeline
    pipeline = RagPipeline(
        embedding_client=OpenAIEmbedding(
            model="text-embedding-3-small"
        ),
        vector_store=MilvusVectorStore(
            host="milvus",
            port=19530,
            collection_name="documents",
            dimension=1536
        ),
        llm_client=OpenAILLM(
            model="gpt-4-turbo"
        )
    )
    
    logger.info("Services initialized")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup."""
    await cache.close()

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint with caching."""
    QUERY_COUNTER.inc()
    start_time = time.time()
    
    try:
        # Check cache
        cached_result = None
        if request.use_cache:
            cache_key = _make_cache_key(request.query, request.limit)
            cached_result = await cache.get(cache_key)
            
            if cached_result:
                CACHE_HITS.inc()
                result = json.loads(cached_result)
                latency = (time.time() - start_time) * 1000
                
                return QueryResponse(
                    **result,
                    cached=True,
                    latency_ms=latency
                )
            
            CACHE_MISSES.inc()
        
        # Query RAG pipeline
        result = await pipeline.query(
            query=request.query,
            limit=request.limit
        )
        
        # Format response
        response_data = {
            "answer": result.answer,
            "sources": [
                {
                    "text": src.text,
                    "metadata": src.metadata,
                    "score": src.score
                }
                for src in result.sources
            ]
        }
        
        # Cache result
        if request.use_cache:
            await cache.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(response_data)
            )
        
        latency = (time.time() - start_time) * 1000
        QUERY_LATENCY.observe(latency / 1000)
        
        return QueryResponse(
            **response_data,
            cached=False,
            latency_ms=latency
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    return generate_latest()

def _make_cache_key(query: str, limit: int) -> str:
    """Generate cache key."""
    content = f"{query}:{limit}"
    return f"rag:{hashlib.md5(content.encode()).hexdigest()}"
```

## Monitoring

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'rag-api'
    static_configs:
      - targets: ['rag-api:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RAG Pipeline Metrics",
    "panels": [
      {
        "title": "Query Rate",
        "targets": [
          {
            "expr": "rate(rag_queries_total[5m])"
          }
        ]
      },
      {
        "title": "Query Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_query_latency_seconds)"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(rag_cache_hits_total[5m]) / (rate(rag_cache_hits_total[5m]) + rate(rag_cache_misses_total[5m]))"
          }
        ]
      }
    ]
  }
}
```

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  rag-api:
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api
```

### Nginx Configuration

```nginx
# nginx.conf
upstream rag_backend {
    least_conn;
    server rag-api-1:8000;
    server rag-api-2:8000;
    server rag-api-3:8000;
    server rag-api-4:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

## Best Practices

### 1. Environment Configuration

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4-turbo"
    embedding_model: str = "text-embedding-3-small"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600
    
    # Performance
    max_workers: int = 4
    request_timeout: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. Error Handling

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler."""
    logger.error(f"Global error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )
```

### 3. Rate Limiting

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.on_event("startup")
async def startup():
    await FastAPILimiter.init(cache)

@app.post("/query", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def query(request: QueryRequest):
    """Rate-limited query endpoint."""
    # ...
```

### 4. Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    "logs/rag.log",
    maxBytes=10_000_000,  # 10MB
    backupCount=5
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler, logging.StreamHandler()]
)
```

## Deployment

### Development

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Stop services
docker-compose down
```

### Production

```bash
# Build and push image
docker build -t myregistry/rag-api:latest .
docker push myregistry/rag-api:latest

# Deploy with scaling
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d

# Update service
docker-compose pull rag-api
docker-compose up -d rag-api
```

### Kubernetes

```yaml
# kubernetes/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 4
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: myregistry/rag-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MILVUS_HOST
          value: "milvus-service"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Performance Tuning

### 1. Batch Processing

```python
async def batch_query(queries: list[str]) -> list[dict]:
    """Process multiple queries in parallel."""
    tasks = [pipeline.query(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results
```

### 2. Connection Pooling

```python
from pymilvus import connections

# Configure connection pool
connections.connect(
    alias="default",
    host="milvus",
    port=19530,
    pool_size=10
)
```

### 3. Caching Strategy

```python
# Multi-level caching
class CacheManager:
    def __init__(self):
        self.memory_cache = {}  # In-memory
        self.redis_cache = redis_client  # Distributed
    
    async def get(self, key: str):
        # Check memory first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check Redis
        value = await self.redis_cache.get(key)
        if value:
            self.memory_cache[key] = value
        
        return value
```

## Security

### 1. API Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/query")
async def query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Protected endpoint."""
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Process query
    # ...
```

### 2. Input Validation

```python
from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    
    @validator('query')
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Limit must be between 1 and 100')
        return v
```

## Next Steps

- [Custom Vector Store](custom_vectorstore.md) - Implement custom stores
- [Hybrid Search](hybrid_search.md) - Advanced search
- [RAG Pipeline Guide](../user_guide/rag_pipeline.md) - Pipeline details

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
