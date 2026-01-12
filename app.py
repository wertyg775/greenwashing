# backend/app.py - Optimized for External Redis (Upstash/Redis Cloud)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from greenwashing_analyzer import GreenwashingAnalyzer
import json
import hashlib
import os
from typing import Optional

# Redis with connection pooling for external services
try:
    import redis
    from redis.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("ERROR: redis package not installed!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTL in seconds (24 hours)
CACHE_TTL = 24 * 60 * 60

# Initialize Redis with connection pooling
REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    raise ValueError(
        "REDIS_URL environment variable is required for Option 2. "
        "Please set it in your HF Spaces secrets or .env file."
    )

# Create connection pool for better performance
pool = ConnectionPool.from_url(
    REDIS_URL,
    max_connections=10,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
    retry_on_timeout=True
)

redis_client = redis.Redis(connection_pool=pool)

# Test connection on startup
try:
    redis_client.ping()
    print("✓ Redis connected successfully")
    print(f"✓ Redis URL: {REDIS_URL.split('@')[1] if '@' in REDIS_URL else 'localhost'}")
except Exception as e:
    print(f"ERROR: Failed to connect to Redis: {e}")
    raise

analyzer = GreenwashingAnalyzer(
    specificity_model_path="./models/specificity"
)

class AnalysisRequest(BaseModel):
    text: str

def get_cache_key(text: str, method: str = "additive") -> str:
    """Generate a unique cache key based on text and analysis method"""
    content = f"{method}:{text}"
    return f"greenwash:{hashlib.md5(content.encode()).hexdigest()}"

@app.post("/api/analyze_additive")
async def analyze_additive(request: AnalysisRequest):
    cache_key = get_cache_key(request.text, "additive")
    
    # Try to get from cache
    try:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            result = json.loads(cached_result)
            result["cached"] = True
            result["cache_source"] = "redis"
            return result
    except redis.RedisError as e:
        print(f"Redis read error: {e}")
        # Continue without cache on error
    
    # Cache miss - perform analysis
    result = analyzer.analyze_additive(request.text)
    result["cached"] = False
    
    # Store in cache with error handling
    try:
        redis_client.setex(
            cache_key,
            CACHE_TTL,
            json.dumps(result)
        )
    except redis.RedisError as e:
        print(f"Redis write error: {e}")
        # Continue even if caching fails
    
    return result

@app.get("/health")
async def health():
    redis_status = "unknown"
    redis_info = {}
    
    try:
        redis_client.ping()
        redis_status = "connected"
        
        # Get Redis server info
        info = redis_client.info()
        redis_info = {
            "version": info.get("redis_version"),
            "uptime_days": info.get("uptime_in_days"),
            "connected_clients": info.get("connected_clients"),
            "used_memory_human": info.get("used_memory_human")
        }
    except redis.RedisError as e:
        redis_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "cache": {
            "status": redis_status,
            "type": "redis_external",
            "info": redis_info
        }
    }

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    try:
        info = redis_client.info()
        keys_count = redis_client.dbsize()
        
        # Get greenwash-specific keys
        greenwash_keys = redis_client.scan_iter("greenwash:*")
        count = sum(1 for _ in greenwash_keys)

        
        return {
            "total_keys": keys_count,
            "greenwash_keys": len(greenwash_keys),
            "cache_type": "redis_external",
            "ttl_hours": CACHE_TTL / 3600,
            "memory": {
                "used": info.get("used_memory_human"),
                "peak": info.get("used_memory_peak_human"),
                "fragmentation_ratio": info.get("mem_fragmentation_ratio")
            },
            "stats": {
                "total_commands": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": round(
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100, 
                    2
                ) if (info.get("keyspace_hits") or info.get("keyspace_misses")) else 0
            }
        }
    except redis.RedisError as e:
        return {"error": str(e)}

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all greenwash cache entries"""
    try:
        keys = redis_client.keys("greenwash:*")
        if keys:
            cleared = redis_client.delete(*keys)
            return {
                "cleared": cleared,
                "cache_type": "redis_external",
                "message": f"Cleared {cleared} cache entries"
            }
        return {
            "cleared": 0,
            "cache_type": "redis_external",
            "message": "No cache entries to clear"
        }
    except redis.RedisError as e:
        return {"error": str(e)}

@app.get("/cache/inspect/{text_hash}")
async def inspect_cache(text_hash: str):
    """Inspect a specific cache entry by hash"""
    try:
        cache_key = f"greenwash:{text_hash}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            ttl = redis_client.ttl(cache_key)
            return {
                "found": True,
                "data": json.loads(cached_result),
                "ttl_seconds": ttl,
                "ttl_hours": round(ttl / 3600, 2)
            }
        return {"found": False, "message": "Cache entry not found"}
    except redis.RedisError as e:
        return {"error": str(e)}

@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection pool on shutdown"""
    try:
        redis_client.close()
        pool.disconnect()
        print("✓ Redis connection closed")
    except Exception as e:
        print(f"Error closing Redis: {e}")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")