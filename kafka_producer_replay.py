
import argparse
import csv
import json
import os
import time

from dotenv import load_dotenv
from kafka import KafkaProducer

load_dotenv()

COLUMNS = ["id", "headline", "url", "author", "date", "ticker"]

producer = KafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def publish_headline(headline: str, stock: str, published_at: str = None):
    payload = {"headline": headline, "stock": stock}
    if published_at:
        payload["published_at"] = published_at
    producer.send("headlines", value=payload)


def stream_csv(path, limit=None, rate=5.0, tickers=None, has_header=False):
    """
    Reads the CSV row by row (never loads the whole 312MB file into memory)
    and publishes one headline every (1 / rate) seconds.
    """
    tickers_filter = set(t.strip().upper() for t in tickers.split(",")) if tickers else None
    sent = 0
    delay = 1.0 / rate if rate > 0 else 0

    with open(path, newline="", encoding="latin-1") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)

        for row in reader:
            if limit is not None and sent >= limit:
                break
            if len(row) < 6:
                continue  # skip malformed rows

            row_dict = dict(zip(COLUMNS, row[:6]))
            headline = row_dict["headline"].strip()
            ticker = row_dict["ticker"].strip().upper()
            date = row_dict["date"].strip()

            if not headline:
                continue
            if tickers_filter and ticker not in tickers_filter:
                continue

            publish_headline(headline, ticker, date)
            sent += 1

            if sent % 50 == 0:
                print(f"Published {sent} headlines so far...")

            if delay:
                time.sleep(delay)

    producer.flush()
    print(f"\nDone. Published {sent} headlines total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the historical headlines CSV")
    parser.add_argument("--limit", type=int, default=None,
                         help="Max number of headlines to publish (omit for all rows)")
    parser.add_argument("--rate", type=float, default=5.0,
                         help="Headlines per second (default: 5). Lower = gentler on CPU.")
    parser.add_argument("--tickers", type=str, default=None,
                         help="Comma-separated tickers to filter by, e.g. AAPL,TSLA")
    parser.add_argument("--has-header", action="store_true",
                         help="Set if the CSV has a header row to skip")
    args = parser.parse_args()

    stream_csv(
        args.file,
        limit=args.limit,
        rate=args.rate,
        tickers=args.tickers,
        has_header=args.has_header,
    )