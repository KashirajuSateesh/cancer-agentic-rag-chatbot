import json
import os
from datetime import datetime


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag_logs.jsonl")


def log_rag_event(event_data):
    os.makedirs(LOG_DIR, exist_ok=True)

    log_record = {
        "timestamp": datetime.now().isoformat(),
        **event_data
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")