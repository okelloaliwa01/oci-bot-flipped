#!/usr/bin/env bash
docker build -t bf_scalper:latest .
docker run --env-file .env -v $(pwd)/data:/app/data bf_scalper:latest
