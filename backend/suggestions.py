"""Prijedlozi sljedećih pitanja na temelju teme trenutnog upita."""

from __future__ import annotations

# Ključne riječi koje pitanje svrstavaju u temu.
_CATEGORY_KEYWORDS = {
    "upisi": ["upis", "prijav", "dokument", "uvjet"],
    "ispiti": ["ispit", "rokovi", "kolokvij", "prijav", "završni", "diplomski rad"],
    "nastava": ["raspored", "predavanje", "nastav", "akademsk", "kalendar", "online"],
    "kontakt": ["kontakt", "e-mail", "email", "telefon", "studentsk", "služb"],
    "skolarina": ["školarin", "plaćanj", "rata", "popust", "stipendij"],
    "studijski_programi": ["program", "fakultet", "studij", "ekonomij", "informatik", "medicin"],
    "sveuciliste_info": ["sveučilišt", "jurja", "dobril", "pul", "adres", "web"],
    "studiranje": ["ECTS", "bod", "godina", "potvrda", "studiranj", "e-učenje"],
}

# Za temu upita -> koje teme nuditi kao sljedeća pitanja.
_RELATED_TOPICS = [
    (["upis", "prijav", "uvjet"], ["upisi", "skolarina", "studijski_programi"]),
    (["ispit", "kolokvij", "završni", "diplomski rad"], ["ispiti", "nastava", "studiranje"]),
    (["raspored", "predavanje", "nastav", "akademsk"], ["nastava", "studiranje", "kontakt"]),
    (["kontakt", "e-mail", "telefon", "služb"], ["kontakt", "sveuciliste_info"]),
    (["školarin", "plaćanj", "stipendij"], ["skolarina", "upisi", "kontakt"]),
    (["program", "fakultet", "studij"], ["studijski_programi", "upisi", "sveuciliste_info"]),
    (["sveučilišt", "jurja", "dobril", "pul"], ["sveuciliste_info", "kontakt", "studijski_programi"]),
    (["ECTS", "bod", "godina", "potvrda"], ["studiranje", "ispiti", "kontakt"]),
]

_DEFAULT_TOPICS = ["sveuciliste_info", "kontakt", "studijski_programi"]

_GENERAL_QUESTIONS = [
    "Koje je službeno ime Sveučilišta u Puli?",
    "Kako mogu kontaktirati studentsku službu?",
    "Gdje mogu pronaći raspored predavanja?",
    "Koji su rokovi za prijavu ispita?",
    "Koje fakultete ima Sveučilište Jurja Dobrile u Puli?",
]


class SuggestionEngine:
    def __init__(self, qa_pairs: list[dict]):
        self.by_category = self._categorize(qa_pairs)

    @staticmethod
    def _categorize(qa_pairs: list[dict]) -> dict[str, list[str]]:
        by_category = {category: [] for category in _CATEGORY_KEYWORDS}
        for qa in qa_pairs:
            question_lower = qa["question"].lower()
            for category, keywords in _CATEGORY_KEYWORDS.items():
                if any(keyword.lower() in question_lower for keyword in keywords):
                    by_category[category].append(qa["question"])
        return by_category

    def next_questions(self, query: str) -> list[str]:
        """Do 3 prijedloga: prvo iz srodnih tema, zatim općenita pitanja."""
        topics = self._topics_for(query.lower())

        suggested = []
        for topic in topics:
            for question in self.by_category.get(topic, []):
                if question != query and question not in suggested:
                    suggested.append(question)
                    break

        for question in _GENERAL_QUESTIONS:
            if len(suggested) >= 3:
                break
            if question != query and question not in suggested:
                suggested.append(question)

        return suggested[:3]

    @staticmethod
    def _topics_for(query_lower: str) -> list[str]:
        for trigger_words, topics in _RELATED_TOPICS:
            if any(word in query_lower for word in trigger_words):
                return topics
        return _DEFAULT_TOPICS
