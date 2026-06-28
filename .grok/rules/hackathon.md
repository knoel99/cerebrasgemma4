# Règles hackathon Gemma 4

## Obligatoire

- Model ID : `gemma-4-31b` sur Cerebras Inference uniquement pour le cœur du produit
- Un second provider (ex. Gemini) est OK uniquement pour benchmark latence côte à côte
- Images : format OpenAI `image_url` avec data URI base64 — pas d'URL hébergée

## API — paramètres par défaut recommandés

```python
model="gemma-4-31b"
stream=True
temperature=1.0   # doc Cerebras ; baisser pour tâches déterministes
top_p=0.95
max_completion_tokens=32768
```

## Pièges connus

- `reasoning_effort` absent = pas de chain-of-thought ; utiliser `none` explicitement si off
- Formats `raw` / `hidden` reasoning non supportés sur Gemma 4
- Endpoint Completions ne supporte pas les images — utiliser Chat Completions
- Rate limits hackathon : ~100 RPM / 100K TPM (après formulaire capacity increase)

## Soumission

- Vidéo ≤ 60s montrant vitesse + features clés
- Masquer clés API, notifications, onglets sensibles à l'enregistrement