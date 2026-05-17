# 05_agentcore/rag.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent))
from config import get_session

S3_BUCKET = os.getenv("S3_KB_BUCKET", "punchh-legacy-development-cost-optimizer-lambda")
S3_PREFIX = os.getenv("S3_KB_PREFIX", "bedrockcore_kb/")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def search_knowledge_base(query: str, top_k: int = 2) -> str:
    """Search S3 markdown files and return relevant context for LLM."""
    print(f"   [RAG] Searching S3: '{query[:60]}'")

    s3 = get_session().client("s3", region_name=AWS_REGION)
    objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX).get("Contents", [])
    query_words = set(query.lower().split())
    results = []

    for obj in objects:
        if not obj["Key"].endswith(".md"):
            continue
        content = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read().decode("utf-8")
        score = sum(1 for w in query_words if w in content.lower())
        if score > 0:
            lines = content.split("\n")
            relevant = []
            for i, line in enumerate(lines):
                if any(w in line.lower() for w in query_words):
                    relevant.extend(lines[max(0, i-1):min(len(lines), i+6)])
                    relevant.append("---")
            results.append({"source": obj["Key"].split("/")[-1], "score": score, "content": "\n".join(relevant[:40])})

    results.sort(key=lambda x: x["score"], reverse=True)
    if not results:
        return "No relevant documentation found."

    print(f"   [RAG] Found {len(results[:top_k])} docs")
    return "\n\n".join(f"[{r['source']}]\n{r['content']}" for r in results[:top_k])


if __name__ == "__main__":
    for q in ["PR merge DBA approval", "TimeOverride business hours"]:
        print(f"\n{'='*50}\nQuery: {q}\n{'='*50}")
        print(search_knowledge_base(q)[:400])