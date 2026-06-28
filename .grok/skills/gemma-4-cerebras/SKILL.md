---
name: gemma-4-cerebras
description: >
  Gemma 4 31B sur Cerebras Inference pour le hackathon. Utiliser pour tout code
  appelant le LLM : chat, streaming, images, tool calling, structured outputs,
  reasoning_effort, ou comparaison de latence. Déclencher sur gemma-4, Cerebras,
  inference, multimodal, hackathon.
---

# Gemma 4 sur Cerebras

Module LLM : `src/cerebrasgemma4/llm.py`. Pas de wrapper SDK supplémentaire sauf besoin explicite.

## Setup

```bash
source venv/bin/activate
pip install -e .
cp .env.example .env   # CEREBRAS_API_KEY
```

## Streaming

```python
from cerebrasgemma4 import stream, MODEL

for token in stream([{"role": "user", "content": "Hello"}]):
    print(token, end="", flush=True)
```

## Image (base64 data URI — seul format supporté)

```python
import base64
from cerebrasgemma4 import stream

b64 = base64.b64encode(open("shot.png", "rb").read()).decode()
msgs = [{"role": "user", "content": [
    {"type": "text", "text": "Describe this."},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
]}]
```

## Tool calling / structured outputs / reasoning

Passer les kwargs OpenAI-compat à `stream()` ou appeler `_client.chat.completions.create` directement.

- Tools : https://inference-docs.cerebras.ai/capabilities/tool-use
- Reasoning : `reasoning_effort="medium"` (off par défaut)
- Démo latence : lire `usage` et `time_info` sur réponse non-stream

## Constantes

- Model ID : `gemma-4-31b` (exporté `MODEL` dans `cerebrasgemma4.llm`)
- Images : max 5, 10 MB total, PNG/JPEG base64 only