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

Pitanja i odgovore uredite u `backend/data/qa_data.json`. Nakon uređivanja ponovno pokrenite backend.
