# Cerebras × Gemma 4 Hackathon

Projet hackathon 24h — Gemma 4 31B sur Cerebras Inference. Grok doit toujours utiliser ce modèle comme LLM principal.

## Contexte hackathon

- **Deadline** : lundi 29 juin 2026, 10h PT
- **Discord** : [#gemma-4-hackathon](https://discord.gg/XWXRquhx7H)
- **Tracks** :
  - Multiverse Agents (multi-agent + multimodal) → `#g4hackathon-multiverse-agents`
  - People's Choice (impressions X) → `#g4hackathon-people-choice` + tag @Cerebras @googlegemma
  - Enterprise Impact → `#g4hackathon-enterprise-impact`
- **Démo vidéo** : max 60s, montrer la vitesse Cerebras (comparaison latence recommandée)
- Doc complète : `docs/Gemma 4 Hackathon.md`

## Stack

- Python 3.12+, venv dans `venv/`
- SDK : `cerebras-cloud-sdk` (API OpenAI-compatible)
- Secrets : `.env` (jamais committer — voir `.env.example`)

## Modèle Cerebras

| Paramètre | Valeur |
|-----------|--------|
| Model ID | `gemma-4-31b` |
| Contexte (hackathon) | 65K MSL / 32K MCL |
| Max output | 32K tokens |
| Images | base64 data URI uniquement (PNG/JPEG), max 5 images, 10 MB total |
| Reasoning | off par défaut — activer via `reasoning_effort`: `low` / `medium` / `high` |
| Capacités | streaming, tool calling, structured outputs (`strict: true`), multimodal |

Docs : https://inference-docs.cerebras.ai/models/gemma-4-31b

## Conventions code

- Code et commentaires en anglais ; réponses utilisateur en français si l'utilisateur parle français
- LLM : `src/cerebrasgemma4/llm.py` (`stream()`, `MODEL`)
- `scripts/chat.py` = smoke test
- Préférer le streaming pour l'UX (vitesse = critère de jugement)
- Exposer `time_info` / `usage` quand pertinent pour les démos de latence
- Ne jamais hardcoder de clé API ni logger de secrets

## Commandes utiles

```bash
source venv/bin/activate
pip install -e .
cp .env.example .env   # puis renseigner CEREBRAS_API_KEY
python scripts/chat.py # smoke test streaming
```

## Priorités développement

1. Gemma 4 sur Cerebras = composant central (pas un provider secondaire)
2. Démontrer la vitesse d'inférence dans l'UX
3. Exploiter le multimodal (images) et/ou tool calling selon le track visé
4. Scaffolding pré-existant OK ; la fonctionnalité core doit être développée pendant le hackathon