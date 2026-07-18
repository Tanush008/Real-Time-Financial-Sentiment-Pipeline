import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Tracked tickers -> search query used against NewsAPI. Extend as needed.
WATCHLIST = {
    "TSLA": "Tesla stock",
    "AAPL": "Apple stock",
    "MSFT": "Microsoft stock",
    "AMZN": "Amazon stock",
    "NVDA": "Nvidia stock",
}

FALLBACK_HEADLINES = [
    ("Tesla stock hits record high after strong earnings", "TSLA"),
    ("Company faces massive losses due to fraud", "XYZ"),
    ("Fed signals rate cuts as inflation cools", "MACRO"),
    ("Apple shares slide on weaker iPhone demand", "AAPL"),
    ("Nvidia beats estimates on booming AI chip demand", "NVDA"),
]

producer = KafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def publish_headline(headline: str, stock: str = "UNKNOWN"):
    payload = {"headline": headline, "stock": stock}
    producer.send("headlines", value=payload)
    print(f"Published: {payload}")


def fetch_live_headlines(page_size: int = 5):
    """Fetch recent headlines per watchlist ticker from NewsAPI."""
    all_articles = []
    for stock, query in WATCHLIST.items():
        try:
            resp = requests.get(
                NEWSAPI_URL,
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            for article in resp.json().get("articles", []):
                title = article.get("title")
                if title:
                    all_articles.append((title, stock))
        except requests.RequestException as e:
            print(f"NewsAPI request failed for {stock}: {e}")
    return all_articles


def run_once():
    if NEWSAPI_KEY:
        articles = fetch_live_headlines()
        if not articles:
            print("NewsAPI returned no articles, falling back to sample headlines.")
            articles = FALLBACK_HEADLINES
    else:
        print("NEWSAPI_KEY not set -- using sample headlines. "
              "Get a free key at https://newsapi.org/register and set it in .env")
        articles = FALLBACK_HEADLINES

    for headline, stock in articles:
        publish_headline(headline, stock)
    producer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=0,
                         help="If set, repeat every N seconds instead of running once.")
    args = parser.parse_args()

    if args.loop:
        while True:
            run_once()
            print(f"Sleeping {args.loop}s until next fetch...")
            time.sleep(args.loop)
    else:
        run_once()
