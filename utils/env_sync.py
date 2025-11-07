# utils/env_sync.py
import os
from dotenv import set_key, dotenv_values

def update_env_file(env_path, params: dict):
    """
    Safely update .env file with optimized params (no duplicates).
    """
    existing = dotenv_values(env_path)
    for k, v in params.items():
        existing[k] = str(v)

    lines = []
    for key, val in existing.items():
        lines.append(f"{key}={val}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"✅ .env updated successfully at {env_path}")
