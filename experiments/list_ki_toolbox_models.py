"""
List models available on the KIT KI-Toolbox (Open WebUI) instance,
and try to flag ones that look like embedding models.

Usage:
    Fill in KIT_TOOLBOX_API_KEY in experiments/.env, then:
    python list_ki_toolbox_models.py
"""
import os
import sys
import json
import urllib.request

from dotenv import load_dotenv

BASE_URL = "https://ki-toolbox.scc.kit.edu"

EMBEDDING_HINTS = ("embed", "embedding", "bge", "gte", "e5-", "nomic-embed", "mxbai")


def main():
    load_dotenv()
    token = os.environ.get("KIT_TOOLBOX_API_KEY")
    if not token:
        print("Set KIT_TOOLBOX_API_KEY in experiments/.env first.", file=sys.stderr)
        sys.exit(1)

    req = urllib.request.Request(
        f"{BASE_URL}/api/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    models = data.get("data", data if isinstance(data, list) else [])

    print(f"Total models: {len(models)}\n")
    embedding_like = []
    for m in models:
        name = m.get("name") or m.get("id", "")
        model_id = m.get("id", "")
        print(f"- {name}  (id: {model_id})")
        if any(h in name.lower() or h in model_id.lower() for h in EMBEDDING_HINTS):
            embedding_like.append((name, model_id))

    print("\nLikely embedding models (heuristic name match):")
    if embedding_like:
        for name, model_id in embedding_like:
            print(f"- {name}  (id: {model_id})")
    else:
        print("None matched by name — the /api/models list may only show chat-facing "
              "models. Check Admin Panel > Settings > Documents (RAG) for the embedding "
              "model actually configured for retrieval.")


if __name__ == "__main__":
    main()
