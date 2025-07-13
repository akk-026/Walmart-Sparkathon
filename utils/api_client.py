import requests

BASE = "http://127.0.0.1:8000"

def rank_products_by_relevance(product_ids, customer_zip):
    payload = {"product_ids": product_ids, "customer_zip": customer_zip}
    try:
        resp = requests.post(
            f"{BASE}/rank-products",
            json=payload,
            timeout=10  # bump from 3s → 10s
        )
        if resp.status_code != 200:
            print("💥 Rank API error:", resp.status_code, resp.text)
        resp.raise_for_status()
        return resp.json().get("ranked_products", [])
    except Exception as e:
        print("⚠️ Exception calling Rank API:", e)
        return []  # fail-safe empty
