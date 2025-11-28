#!/bin/bash
set -e
echo "Starting Ollama server"
/bin/ollama serve &

OLLAMA_PID=$!

echo "Waiting for ollama to be ready"
while ! ollama list | grep -q 'NAME';do
    sleep 1
done

echo "Pulling models : nomic-embed-text and llama3"
ollama pull nomic-embed-text
ollama pull llama3

echo "✅ Ollama models are ready!"

# Wait for the background process (ollama serve) to keep the container alive
wait $OLLAMA_PID
