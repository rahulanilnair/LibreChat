#!/bin/bash

# 1. Start Ollama in the background.
/bin/ollama serve &
pid=$!

# 2. Wait for Ollama to wake up.
echo "🔴 Waiting for Ollama to start..."

# We use 'ollama list' to check if the server is responding.
# If the server is down, this command will fail (exit code 1).
while ! ollama list > /dev/null 2>&1; do
    sleep 1
done

echo "🟢 Ollama is running!"

# 3. Pull the models.
echo "⬇️  Pulling nomic-embed-text..."
ollama pull nomic-embed-text

echo "⬇️  Pulling llama3..."
ollama pull llama3.2

# 4. Wait for the server process to finish (keeps the container alive).
wait $pid