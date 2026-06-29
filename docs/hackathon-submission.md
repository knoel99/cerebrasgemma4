# Soumission hackathon — Cerebras × Gemma 4

Textes prêts à copier-coller pour Discord (3 tracks) et X.  
Deadline : **lundi 29 juin 2026, 10h PT**.

Liens à renseigner avant envoi :

| Placeholder | Valeur |
|-------------|--------|
| `VIDEO_URL` | URL de la démo enregistrée (≤ 60 s) |
| `DEMO_URL` | URL de l'app live (si hébergée) |
| `REPO_URL` | https://github.com/knoel99/cerebrasgemma4 |

---

## Nom d'application — recommandation

### Problème avec *Vid2Doc*

- Sonne **utilitaire** (convertisseur) plutôt que produit.
- Le « 2 » fait daté ; ne dit rien sur la **vitesse**, le **multimodal** ni le **chat**.
- Difficile à mémoriser sur X et en jugement « People's Choice ».

### Nom recommandé : **Sightline**

| Critère | Pourquoi ça marche |
|---------|-------------------|
| **Sens** | *Sight* (vision multimodale) + *line* (fil narratif, timestamps, chapitres) |
| **Positionnement** | Pro, crédible pour Enterprise Impact |
| **Mémorabilité** | Un mot, distinctif, facile à hashtag `#Sightline` |
| **Tagline** | *Your sightline into any recording.* |

**Variante orientée vitesse (X / People's Choice)** : **FlashBrief** — *flash inference → structured brief*. Plus punchy pour la démo tok/s, moins « enterprise ».

**À éviter pour la soumission** : *Keyframe* (collision avec l’édition vidéo), *VidScribe* / *VideoBrief* (génériques, noyés dans le bruit).

> **Statut** : l’app et le code utilisent désormais **Sightline** (UI, exports, API). Le repo GitHub reste `cerebrasgemma4`.

---

## Checklist avant envoi

- [ ] Vidéo ≤ **60 s**, vitesse Cerebras visible (tok/s, panneau métriques)
- [ ] Comparaison latence côte à côte (recommandé)
- [ ] Aucune clé API, email, notification visible
- [ ] **3 posts Discord séparés** (un par channel)
- [ ] Post X avec **@Cerebras** et **@googlegemma**
- [ ] Liens `VIDEO_URL` / `DEMO_URL` / `REPO_URL` renseignés

### Structure vidéo suggérée (60 s)

| Temps | Contenu |
|-------|---------|
| 0–5 s | Hook : URL YouTube → rapport structuré en streaming |
| 5–25 s | Pipeline live : Scout → Analyze → Compose + **tok/s** dans le header |
| 25–40 s | Chat « Ask your video » + enrichissement du rapport |
| 40–55 s | Métriques (wall clock vs inference Cerebras) ; comparaison GPU si possible |
| 55–60 s | Export MD/PDF + `gemma-4-31b` / Cerebras Inference |

---

## Track 1 — `#g4hackathon-multiverse-agents`

```
🎬 **Sightline** — Your sightline into any recording, powered by Gemma 4 31B on Cerebras Inference

**Demo video:** VIDEO_URL

**What it does**
Paste a YouTube URL or drop an MP4 → get a Markdown report with executive summary, timestamped sections, embedded key-frame screenshots, and key takeaways. Then chat with the video to Q&A or enrich the report live.

**Why it's multi-agent + multimodal**
Sightline runs a coordinated Gemma 4 pipeline — each stage is a specialized agent:

1. **Scout** — hierarchical multimodal scoring (frame mosaics + structured JSON) to find the most document-worthy moments across long videos
2. **Analyst** — deep per-frame vision analysis (on-screen text, UI, diagrams)
3. **Composer** — streams the final Markdown document token-by-token
4. **Chat agent** — grounded Q&A on transcript + frame analyses, with structured enrichments appended to the report

**Multimodal stack**
- Video → sparse frame extraction (YouTube chapters aware)
- Images sent as base64 data URIs to Gemma 4 via Cerebras Chat Completions
- Structured outputs (`strict: true`) for scout scores and chat responses
- Full transcript fusion for context

**Speed in action**
Cerebras inference is the UX — streaming compose, live tok/s in the header, per-call `time_info` breakdown (TTFT, tokens/s). On a 30+ min video, the full pipeline completes in under 60 seconds in demo mode — impossible without Cerebras-class throughput across 10+ multimodal calls.

**Stack**
Python · FastAPI · `cerebras-cloud-sdk` · `gemma-4-31b`

**Repo:** REPO_URL
**Live demo:** DEMO_URL
```

---

## Track 2 — `#g4hackathon-people-choice`

```
🏆 **Sightline** — People's Choice submission

**Demo video:** VIDEO_URL

Stop scrubbing through hour-long recordings. Sightline turns any video into a polished, shareable document in seconds — powered by **Gemma 4 31B** on **Cerebras Inference**.

**The wow moment**
Watch the report stream in real time while the header shows live **tokens/sec**. A 45-minute tech talk → structured doc with screenshots, timestamps, and takeaways. Then ask "What was on screen at 12:30?" and get a grounded answer instantly.

**Built for the hackathon**
- Multi-agent pipeline (Scout → Analyze → Compose → Chat)
- Multimodal frame mosaics + structured outputs
- Sub-60s end-to-end demo mode on long YouTube videos
- Export .md / .html / .pdf + metrics JSON for latency proof

**Try it:** DEMO_URL
**Code:** REPO_URL

👉 Also posted on X — please like & repost to help us compete for People's Choice! 🙏
```

---

## Track 3 — `#g4hackathon-enterprise-impact`

```
🏢 **Sightline** — Enterprise Impact submission

**Demo video:** VIDEO_URL

**The enterprise problem**
Teams lose institutional knowledge buried in webinars, training recordings, incident post-mortems, and customer calls. Manual documentation is slow, inconsistent, and doesn't scale.

**The solution**
Sightline automatically converts video into searchable, structured knowledge documents — with timestamped sections, embedded screenshots of slides/diagrams/UI, and an executive summary. Employees can then chat with the video context to extract decisions, FAQs, or compliance-relevant details.

**Enterprise use cases**
- **Knowledge management** — onboarding docs from internal training videos
- **Customer support** — product demo → self-service help articles
- **Incident response** — post-mortem recordings → structured timelines with visual evidence
- **Multimodal RAG** — transcript + vision analysis fused into a single grounded context

**Production readiness**
- Rate-limited pipeline (100 RPM / 100K TPM) with progress + ETA
- Job store with resumable state, context persistence for chat
- Export pipeline (MD, HTML, PDF) + metrics JSON for observability
- Configurable report prompts for domain-specific outputs
- Base64 data URIs only — API-compliant, no hosted image URLs

**AI differentiation**
Cerebras speed makes multi-call multimodal pipelines viable in production: 10+ Gemma 4 vision calls + streaming compose in interactive time. Without inference at this throughput, the scout→analyze→compose workflow would be minutes, not seconds.

**Stack:** Python · FastAPI · `gemma-4-31b` on Cerebras Inference

**Repo:** REPO_URL
**Demo:** DEMO_URL
```

---

## X (Twitter) — obligatoire pour Track 2

### Post principal (~280 caractères)

```
Built Sightline for the @Cerebras × @googlegemma hackathon 🚀

Any video → structured Markdown doc in seconds with Gemma 4 31B

Multi-agent: Scout → Analyze → Compose → Chat
Live tok/s · multimodal frame scoring · export PDF

45-min talk → full report in <60s

Demo 👇
VIDEO_URL

#Gemma4 #Cerebras #AI #Multimodal
```

### Thread optionnel (plus de reach)

**Tweet 1**
```
We turned Gemma 4 31B on @Cerebras into a video-to-document factory 🎬→📄

Sightline: paste a YouTube URL, get a structured report with screenshots + timestamps — then chat with your video.

Built in 24h for the @googlegemma hackathon. Demo below 👇
```

**Tweet 2**
```
4 specialized agents, 1 model:

🔍 Scout — scores frame mosaics (structured JSON)
🔬 Analyst — deep vision per key frame
✍️ Composer — streams Markdown live
💬 Chat — grounded Q&A + report enrichments

All on gemma-4-31b via Cerebras Inference.
```

**Tweet 3**
```
Speed IS the feature.

Header shows live tokens/sec. Pipeline metrics expose TTFT per call.

On a 30+ min video: full scout → analyze → compose in under 60 seconds.

Without Cerebras throughput, this UX doesn't exist.

VIDEO_URL
```

**Tweet 4**
```
Enterprise angle: webinars, training, incident recordings → searchable knowledge docs.

Export .md / .html / .pdf. Custom prompts. Rate-limited for production.

Repo: REPO_URL

If this is useful, a repost helps us for People's Choice 🙏
@Cerebras @googlegemma
```

### Variante courte « FlashBrief » (si tu veux pousser la vitesse sur X)

```
FlashBrief ⚡ — 45 min of video → full structured doc in <60s

Gemma 4 31B on @Cerebras: Scout, Analyze, Compose, Chat — all streaming, live tok/s on screen.

@googlegemma hackathon entry 👇
VIDEO_URL
```

---

## Rappel règlement

- **1 post Discord par track** ; mise à jour possible jusqu'à la deadline
- Track 2 : **même soumission Discord + post X** avec @Cerebras @googlegemma
- Vidéo : max **60 s**, montrer la **vitesse Cerebras**
- Modèle central : **`gemma-4-31b`** sur Cerebras Inference

Discord : https://discord.gg/XWXRquhx7H