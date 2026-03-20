import requests

# Call the local Ollama API
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "qwen2.5:7b",
    "prompt": "Explain what a transformer is in simple terms",
    "stream": False
})

print(response.json()["response"])