# Sightline

**Your sightline into any recording** — hackathon project powered by Gemma 4 31B on [Cerebras Inference](https://inference-docs.cerebras.ai/models/gemma-4-31b).

## License

**Temporary proprietary license** (June 28 – July 15, 2026). All rights reserved — no copying, forking, redistribution, or commercial use without permission. See [LICENSE](LICENSE). Previous Apache 2.0 terms are preserved in [LICENSE.apache-2.0](LICENSE.apache-2.0) for post-hackathon relicensing.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env   # renseigner CEREBRAS_API_KEY
python scripts/chat.py # smoke test streaming
```

## Structure

```
cerebrasgemma4/
├── src/cerebrasgemma4/   # package Python
│   ├── __init__.py
│   └── llm.py            # stream(), MODEL
├── scripts/
│   └── chat.py           # smoke test
├── tests/
├── docs/
│   └── Gemma 4 Hackathon.md
├── pyproject.toml
├── requirements.txt
└── AGENTS.md
```

| Chemin | Rôle |
|--------|------|
| `src/cerebrasgemma4/llm.py` | Appels Gemma 4 (`stream()`, `MODEL`) |
| `scripts/chat.py` | Smoke test |
| `.grok/skills/gemma-4-cerebras/` | Skill Grok pour appels LLM |
| `docs/Gemma 4 Hackathon.md` | Doc officielle du hackathon |

## Modèle

- **Model ID** : `gemma-4-31b`
- Multimodal (texte + images base64), streaming, tool calling, structured outputs
- Reasoning off par défaut — `reasoning_effort` pour activer

## Hackathon

- Deadline : lundi 29 juin 2026, 10h PT
- Discord : [#gemma-4-hackathon](https://discord.gg/XWXRquhx7H)
- Démo vidéo ≤ 60s montrant la vitesse Cerebras