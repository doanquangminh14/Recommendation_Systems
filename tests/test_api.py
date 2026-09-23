import os
import sys

# Reconfigure stdout for UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.main import app


def test_api_endpoints():
    with TestClient(app) as client:
        print("--- Test 1: Health Check ---")
        resp = client.get("/health")
        assert resp.status_code == 200, f"Health failed: {resp.text}"
        data = resp.json()
        print("Health status:", data["status"])
        print("Models status:", data["models_status"])

        print("\n--- Test 2: System Stats ---")
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        print(f"Total games: {data['total_games']}, Total users: {data['total_users']}")

        print("\n--- Test 3: Categories ---")
        resp = client.get("/api/categories")
        assert resp.status_code == 200
        cats = resp.json()
        print(f"Categories count: {len(cats)}, Sample: {cats[:3]}")

        print("\n--- Test 4: Personalized Recommendation ---")
        payload = {
            "user_id": "A100WO06OIG7KW",
            "top_k": 3,
            "use_mmr": True,
            "diversity_lambda": 0.7,
            "include_explanations": True
        }
        resp = client.post("/api/recommend/personalized", json=payload)
        assert resp.status_code == 200, f"Personalized failed: {resp.text}"
        data = resp.json()
        print(f"Strategy: {data['strategy']}, Count: {data['count']}, ILD: {data.get('diversity_ild')}")
        for r in data["recommendations"]:
            print(f"  - [{r['rank']}] {r['title']} (Score: {r['hybrid_score']}, Poster: {bool(r.get('image_url'))})")

        print("\n--- Test 5: Semantic Cold-Start Recommendation ---")
        payload = {
            "query": "zelda adventure open world fantasy exploration",
            "top_k": 3
        }
        resp = client.post("/api/recommend/semantic", json=payload)
        assert resp.status_code == 200, f"Semantic failed: {resp.text}"
        data = resp.json()
        print(f"Semantic Strategy: {data['strategy']}, Count: {data['count']}")
        for r in data["recommendations"]:
            print(f"  - {r['title']} (Score: {r['cb_score']})")

        print("\n--- Test 6: Similar Games Discovery ---")
        item_id = "B00004SVVJ"
        resp = client.post("/api/recommend/similar", json={"item_id": item_id, "top_k": 3})
        assert resp.status_code == 200, f"Similar failed: {resp.text}"
        data = resp.json()
        print(f"Similar items count: {data['count']}")

        print("\n--- Test 7: Explanation ---")
        resp = client.post("/api/explain", json={"user_id": "A100WO06OIG7KW", "item_id": "B00004SVVJ"})
        assert resp.status_code == 200, f"Explain failed: {resp.text}"
        data = resp.json()
        print(f"Explain status: {data['status']}, Anchor: {data.get('anchor_game')}, Reasons: {len(data.get('key_reasons', []))}")

        print("\n--- Test 8: User Analytics ---")
        resp = client.get("/api/user/A100WO06OIG7KW/analytics")
        assert resp.status_code == 200, f"Analytics failed: {resp.text}"
        data = resp.json()
        print(f"Persona: {data['gamer_persona']}, Total reviews: {data['total_interactions']}, Avg rating: {data['average_rating']}")

        print("\n--- Test 9: AI Agent Chat Turn ---")
        resp = client.post("/api/agent/chat", json={"message": "Gợi ý cho tôi game hay", "user_id": "A100WO06OIG7KW"})
        assert resp.status_code == 200, f"Agent chat failed: {resp.text}"
        data = resp.json()
        print(f"Agent Intent: {data['intent']}, Tool used: {data['tool_used']}")
        print("Agent Response snippet:", repr(data["response_text"][:100]) + "...")

        print("\n--- Test 10: Catalog Search ---")
        resp = client.get("/api/search?q=Mario&limit=3")
        assert resp.status_code == 200
        data = resp.json()
        print(f"Search Mario total found: {data['total_found']}, Returned: {len(data['items'])}")

        print("\n--- Test 11: Game Detail ---")
        resp = client.get("/api/games/B00004SVVJ")
        assert resp.status_code == 200
        data = resp.json()
        print(f"Game Detail: {data['title']}, Store: {data.get('store')}, Price: {data.get('price')}")

        print("\n=========================================")
        print("🎉 ALL 11 API ENDPOINTS PASSED PERFECTLY!")
        print("=========================================")


if __name__ == "__main__":
    test_api_endpoints()
