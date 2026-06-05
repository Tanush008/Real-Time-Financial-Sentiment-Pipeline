from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def publish_headline(headline: str, stock: str = "UNKNOWN"):
    payload = {"headline": headline, "stock": stock}
    producer.send('headlines', value=payload)
    producer.flush()
    print(f"Published: {payload}")

# test
if __name__ == "__main__":
    publish_headline("Tesla stock hits record high after strong earnings", "TSLA")
    publish_headline("Company faces massive losses due to fraud", "XYZ")