# sembot

Chatbot koji koristi Semantic Router i FastAPI za odgovaranje na upite studenata o sveučilištu (prvenstveno o upisima).

## Opis

Backend koristi FastAPI za API endpointe i semantičko pretraživanje (HuggingFace encoder) za pronalaženje najboljeg odgovora na temelju sličnosti pitanja. Frontend je jednostavno chat sučelje u HTML/CSS/JavaScript.

## Struktura projekta

```
sembot-v2/
├── Makefile               # Komande za pokretanje projekta
├── requirements.txt       # Python ovisnosti
├── backend/
│   ├── main.py            # FastAPI glavna aplikacija
│   ├── router.py          # Semantic Router implementacija
│   └── data/
│       └── qa_data.json   # Podaci pitanja i odgovora
└── frontend/
    ├── index.html
    └── static/
        ├── css/style.css
        └── js/
            ├── config.js   # URL backend API-ja
            └── script.js
```

## Preduvjeti

- Python 3.8+

## Postavljanje

```
make setup
```

## Pokretanje

Backend i frontend zajedno:

```
make run
```

Samo backend:

```
make run-backend
```

Samo frontend:

```
make run-frontend
```

Pristup aplikaciji:

- Backend: http://localhost:8000
- Frontend: http://localhost:8080

## Prilagodba pitanja i odgovora

Pitanja i odgovore uredite u `backend/data/qa_data.json`. Svaki par može imati i
dodatne formulacije za bolji recall:

```json
{
  "question": "Koji su rokovi za prijavu ispita?",
  "answer": "...",
  "paraphrases": [
    "Do kada se prijavljuju ispiti?",
    "Kada je rok za prijavu na ispitni rok?"
  ]
}
```

Embeddingi se računaju jednom i keširaju u `backend/data/.embeddings_cache.npz`;
cache se automatski osvježi kad se promijene podaci, model ili prefiksi.

Podaci se validiraju pri učitavanju (prazni/dupli unosi ruše start s jasnom
porukom). Prije rasta baze provjerite sumnjivo slične parove (duplikate/konflikte):

```
make check-data            # prag 0.90
python -m backend.check_data 0.85   # stroži prag
```

Health provjera (za Cloud Run / Railway probe): `GET /health` vraća status,
broj QA parova i je li reranker već učitan.

## Konfiguracija (env varijable)

| Varijabla | Default | Opis |
|---|---|---|
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | bi-encoder za pretragu |
| `USE_RERANKER` | `true` | uključi cross-encoder reranker |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | multijezični reranker |
| `MIN_RETRIEVAL_SCORE` | `0.80` | prag ispod kojeg je pitanje izvan domene |
| `RERANK_THRESHOLD` / `RERANK_MARGIN` | `0.50` / `0.15` | pragovi za suzdržavanje |

> Pragovi su početne vrijednosti — kalibrirajte ih na testnom skupu stvarnih upita.
> Pri prvom pokretanju modeli se preuzimaju (e5-base ~440 MB, reranker ~2.3 GB).

## Deploy (Google Cloud Run)

`Dockerfile` zapeče modele i embedding cache u image, pa cold start ne preuzima
ništa s interneta. Reranker se učitava u pozadini, pa prvi upit ne čeka cijeli model.

```
gcloud run deploy sembot-backend \
  --source . \
  --memory 4Gi --cpu 2 --cpu-boost \
  --port 8080 \
  --region europe-west1 \
  --allow-unauthenticated
```

- `--memory 4Gi` — modeli traže prostora u RAM-u (inače OOM pri startu).
- `--cpu-boost` — više CPU-a tijekom starta = brže učitavanje modela.
- Bez `--min-instances` (scale-to-zero): nema fiksnog troška, ali prvi upit nakon
  mirovanja čeka ~5–10 s da se digne kontejner. Za nulti cold start dodajte
  `--min-instances 1` (stalni trošak jedne instance).
