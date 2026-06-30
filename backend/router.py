import json
import os
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

class ChatbotRouter:
    def __init__(self, threshold: float = None):
        # Učitaj podatke pitanja i odgovora
        self.load_qa_data()
        # Model name from env or default
        model_name = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
        print(f"Inicijalizacija HuggingFace encodera: {model_name}")
        self.encoder = HuggingFaceEncoder(model_name=model_name)
        print("HuggingFace encoder inicijaliziran.")
        print("Kreiranje ruta...")
        self.routes = []
        for qa in self.qa_data["qa_pairs"]:
            self.routes.append(
                Route(
                    name=f"q_{len(self.routes)}",
                    utterances=[qa["question"]],
                    response=qa["answer"]
                )
            )
        print(f"Kreirano {len(self.routes)} ruta.")
        # Uklanjamo RouteLayer jer koristimo manual threshold routing
        self.categorize_questions()
        # Threshold from env or default
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = float(os.environ.get("ROUTER_THRESHOLD", 0.6))

    def get_response_sync(self, query: str, threshold: float = None, skip_university_check: bool = False) -> dict:
        """
        Sinkrona verzija get_response za batch evaluaciju (bez async) s ručnim pragom udaljenosti
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        if threshold is None:
            threshold = self.threshold
        if not skip_university_check and not self.is_university_related(query):
            return self.get_default_response()

        try:
            # Enkodiraj query i uzmi samo prvi vektor ako je višedimenzionalan
            query_vec = self.encoder(query)
            query_vec = np.array(query_vec)
            if query_vec.ndim > 1:
                query_vec = query_vec[0]  # Uzmi prvi red ako je 2D
            
            # Enkodiraj sve rute i uzmi prvi vektor ako je višedimenzionalan
            route_vecs = []
            for i, route in enumerate(self.routes):
                rv = self.encoder(route.utterances[0])
                rv = np.array(rv)
                if rv.ndim > 1:
                    rv = rv[0]  # Uzmi prvi red ako je 2D
                route_vecs.append(rv)
            
            # Sada su svi vektori 1D, mogu sigurno stvoriti 2D array
            query_vec = query_vec.reshape(1, -1)
            route_vecs = np.array(route_vecs)  # Sada neće biti problema s oblicima
            
            # Izračunaj cosine distance i pretvori u poboljšanu similarity
            from sklearn.metrics.pairwise import cosine_distances
            import re
            
            # Preprocess funkcija
            def preprocess_text(text):
                # Ukloni interpunkciju i pretvori u mala slova
                text = re.sub(r'[^\w\s]', '', text.lower())
                text = re.sub(r'\s+', ' ', text.strip())
                return text
            
            # Preprocess query i routes za bolje podudaranje
            processed_query = preprocess_text(query)
            processed_routes = [preprocess_text(route.utterances[0]) for route in self.routes]
            
            # Ponovno enkodiranje s preprocessed tekstom
            query_vec_processed = self.encoder(processed_query)
            query_vec_processed = np.array(query_vec_processed)
            if query_vec_processed.ndim > 1:
                query_vec_processed = query_vec_processed[0]
            query_vec_processed = query_vec_processed.reshape(1, -1)
            
            route_vecs_processed = []
            for processed_route in processed_routes:
                rv = self.encoder(processed_route)
                rv = np.array(rv)
                if rv.ndim > 1:
                    rv = rv[0]
                route_vecs_processed.append(rv)
            route_vecs_processed = np.array(route_vecs_processed)
            
            # Cosine distance -> similarity s jednostavnim poboljšanjima
            cosine_distances_scores = cosine_distances(query_vec_processed, route_vecs_processed)[0]
            base_similarities = 1 - cosine_distances_scores
            
            # Minimalni keyword bonus (samo 5% za točne riječi)
            query_words = set(processed_query.split())
            keyword_bonuses = []
            for route_text in processed_routes:
                route_words = set(route_text.split())
                intersection = len(query_words.intersection(route_words))
                if intersection > 0:
                    keyword_bonus = intersection * 0.01  # 1% po zajedničkoj riječi
                else:
                    keyword_bonus = 0
                keyword_bonuses.append(keyword_bonus)
            
            # Finalni rezultat: base similarity + mali keyword bonus
            similarities = base_similarities + np.array(keyword_bonuses)
            
            # Ograniči na maksimalno 1.0
            similarities = np.clip(similarities, 0, 1.0)
            
        except Exception as e:
            print(f"Error in encoding: {e}")
            # Vrati default odgovor ako se dogodi greška
            return {
                "text": "Žao mi je, ne mogu precizno odgovoriti na to pitanje. Za dodatne informacije, možete kontaktirati studentsku službu na ured-za-studente@unipu.hr ili telefonom na 052/377-006.",
                "suggested_questions": [
                    "Kako mogu kontaktirati studentsku službu?",
                    "Gdje mogu pronaći akademski kalendar?",
                    "Koja je web stranica Sveučilišta Jurja Dobrile?"
                ]
            }
        
        # Pronađi najbolji rezultat koristeći poboljšane similarities
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]
        if best_score >= threshold:
            # Pronađi odgovor iz qa_data na temelju best_idx
            best_question = self.routes[best_idx].utterances[0]
            answer = None
            for qa in self.qa_data["qa_pairs"]:
                if qa["question"] == best_question:
                    answer = qa["answer"]
                    break
            
            if answer:
                suggested_questions = self.generate_next_questions(query, answer)
                return {"text": answer, "suggested_questions": suggested_questions}
            else:
                # Fallback ako ne pronađemo odgovor
                return {
                    "text": "Žao mi je, ne mogu precizno odgovoriti na to pitanje. Za dodatne informacije, možete kontaktirati studentsku službu na ured-za-studente@unipu.hr ili telefonom na 052/377-006.",
                    "suggested_questions": [
                        "Kako mogu kontaktirati studentsku službu?",
                        "Gdje mogu pronaći akademski kalendar?",
                        "Koja je web stranica Sveučilišta Jurja Dobrile?"
                    ]
                }
        else:
            return {
                "text": "Žao mi je, ne mogu precizno odgovoriti na to pitanje. Za dodatne informacije, možete kontaktirati studentsku službu na ured-za-studente@unipu.hr ili telefonom na 052/377-006.",
                "suggested_questions": [
                    "Kako mogu kontaktirati studentsku službu?",
                    "Gdje mogu pronaći akademski kalendar?",
                    "Koja je web stranica Sveučilišta Jurja Dobrile?"
                ]
            }
    
    def load_qa_data(self):
        """Učitaj podatke pitanja i odgovora iz lokalne datoteke."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        qa_path = os.path.join(script_dir, "data", "qa_data.json")
        print(f"Učitavanje podataka iz: {qa_path}")
        with open(qa_path, 'r', encoding='utf-8') as file:
            self.qa_data = json.load(file)
        print(f"Učitano {len(self.qa_data['qa_pairs'])} pitanja i odgovora iz lokalne datoteke.")
    
    def categorize_questions(self):
        """Kategorizira pitanja po temama za predviđanje sljedećih pitanja"""
        self.categories = {
            "upisi": [],
            "ispiti": [], 
            "nastava": [],
            "kontakt": [],
            "skolarina": [],
            "studijski_programi": [],
            "sveuciliste_info": [],
            "studiranje": []
        }
        
        # Ključne riječi za kategorizaciju
        category_keywords = {
            "upisi": ["upis", "prijav", "dokument", "uvjet"],
            "ispiti": ["ispit", "rokovi", "kolokvij", "prijav", "završni", "diplomski rad"],
            "nastava": ["raspored", "predavanje", "nastav", "akademsk", "kalendar", "online"],
            "kontakt": ["kontakt", "e-mail", "email", "telefon", "studentsk", "služb"],
            "skolarina": ["školarin", "plaćanj", "rata", "popust", "stipendij"],
            "studijski_programi": ["program", "fakultet", "studij", "ekonomij", "informatik", "medicin"],
            "sveuciliste_info": ["sveučilišt", "jurja", "dobril", "pul", "adres", "web"],
            "studiranje": ["ECTS", "bod", "godina", "potvrda", "studiranj", "e-učenje"]
        }
        
        # Kategoriziraj pitanja
        for qa in self.qa_data["qa_pairs"]:
            question = qa["question"].lower()
            
            # Provjeri za svaku kategoriju
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in question:
                        self.categories[category].append(qa["question"])
                        break
    
    def is_university_related(self, query: str) -> bool:
        """
        Provjerava je li korisničko pitanje vezano za sveučilište.
        
        Args:
            query (str): Korisničko pitanje
            
        Returns:
            bool: True ako je pitanje vezano uz sveučilište, False inače
        """
        # Ključne riječi specifične za akademski kontekst i Sveučilište Jurja Dobrile u Puli
        university_keywords = [
            # Osnovne riječi za komunikaciju
            "bok", "pozdrav", "dobar", "dan", "večer", "jutro", "hej", "halo", "ćao",
            "hvala", "molim", "pomoć", "možeš", "znaš", "reci", "pitanje", "odgovor",
            "tko", "što", "gdje", "kako", "kada", "zašto", "koliko", "koji", "kojim",
            
            # Opći akademski termini (korijeni riječi)
            "fakultet", "sveučiliš", "studij", "profesor", "predavanj", "ispit", "kolokvij",
            "rokov", "nastav", "upis", "predmet", "semest", "akademsk", "diplom",
            "mentor", "student", "knjižnic", "dom", "x-ica", "iksic", "stipendij", "ects",
            "bologn", "dekan", "referad", "učionic", "demonstrat", "asistent",
            
            # Specifični za Sveučilište u Puli
            "pula", "jurj", "dobril", "unipu", "studomat", "školarin", "potvrd", "informatik",
            "ekonomij", "medicin", "turizam", "glazb", "muzičk", "online", "kalendar",
            
            # Dodatni akademski pojmovi
            "završni", "rad", "teza", "disertacij", "obran", "kontakt", "ured", "služb",
            "erasmus", "razmjen", "zagrebačk", "demonstratur", "skript", "literatur", "udžbenik", 
            "indeks", "isvu", "aai", "seminar", "esej", "projekt", "ocjen", "bodov", "koleg", "godin",
            "tema", "istraživanj", "bibliografij", "citiranj", "referenc", "metodologij",
            "apsolvent", "brucoš", "alumni", "prijavnic", "upitnik", "studiranj", "apsolvir",
            "stranic", "prijav", "status", "uvjet", "pisat", "napisat",
            "autor", "nastavnik", "konzultacij", "program", "obvez", "zbornic", "menz", "blagavaon", 
            "prijemni", "bodovanj", "natječaj", "rang", "list", "kvot", "prag"
        ]
        
        # Pretvori u mala slova radi lakše usporedbe
        query_lower = query.lower()
        
        # Provjeri sadrži li pitanje bilo koju ključnu riječ vezanu za sveučilište
        for keyword in university_keywords:
            if keyword.lower() in query_lower:
                return True
        
        # Dodatna provjera za česte fraze vezane uz studiranje koje ne sadrže nužno ključne riječi
        common_phrases = [
            "kako napisati", "kako predati", "kako dobiti", "kako se prijaviti", 
            "gdje predati", "gdje pronaći", "koji su uvjeti", "koji su rokovi",
            "kada je rok", "kada počinje", "kada završava", "što trebam",
            "što moram", "trebam li", "moram li", "mogu li", "gdje je",
            "koja je procedura", "kako funkcionira", "kako radi"
        ]
        
        # Ako ima neku od čestih fraza, vjerojatno je vezano za sveučilište u kontekstu chatbota
        for phrase in common_phrases:
            if phrase in query_lower:
                return True
        
        return False

    def get_default_response(self) -> dict:
        """
        Vraća standardni odgovor kada korisnik postavi pitanje izvan teme sveučilišta.
        
        Returns:
            dict: Standardni odgovor s prijedlogom za povratak na temu sveučilišta
        """
        default_response = {
            "text": (
                "Izgleda da je vaše pitanje izvan okvira informacija o Sveučilištu Jurja Dobrile u Puli. "
                "Ja sam specijaliziran za pružanje informacija o studijskim programima, rokovima, "
                "događanjima na fakultetu, postupcima upisa i drugim temama vezanim uz Sveučilište. "
                "Možete me pitati o rasporedu nastave, kontakt informacijama, rokovima za predaju radova "
                "ili bilo čemu drugom vezanom uz studiranje na Sveučilištu u Puli."
            ),
            "suggested_questions": [
                "Kako mogu kontaktirati studentsku službu?",
                "Gdje mogu pronaći raspored predavanja?",
                "Koji su rokovi za prijavu ispita?"
            ]
        }
        return default_response
    
    def generate_next_questions(self, query: str, answer: str) -> list:
        """
        Generira prijedloge za sljedeća pitanja na temelju trenutnog pitanja i odgovora.
        
        Args:
            query (str): Trenutno korisničko pitanje
            answer (str): Odgovor na trenutno pitanje
            
        Returns:
            list: Lista prijedloga za sljedeća pitanja
        """
        # Pretvori u mala slova radi lakše usporedbe
        query_lower = query.lower()
        
        # Definiraj tipove pitanja koji bi mogli slijediti trenutno
        related_categories = []
        
        # Provjeri kojem tipu pripada trenutno pitanje
        if any(word in query_lower for word in ["upis", "prijav", "uvjet"]):
            related_categories = ["upisi", "skolarina", "studijski_programi"]
        elif any(word in query_lower for word in ["ispit", "kolokvij", "završni", "diplomski rad"]):
            related_categories = ["ispiti", "nastava", "studiranje"]
        elif any(word in query_lower for word in ["raspored", "predavanje", "nastav", "akademsk"]):
            related_categories = ["nastava", "studiranje", "kontakt"]
        elif any(word in query_lower for word in ["kontakt", "e-mail", "telefon", "služb"]):
            related_categories = ["kontakt", "sveuciliste_info"]
        elif any(word in query_lower for word in ["školarin", "plaćanj", "stipendij"]):
            related_categories = ["skolarina", "upisi", "kontakt"]
        elif any(word in query_lower for word in ["program", "fakultet", "studij"]):
            related_categories = ["studijski_programi", "upisi", "sveuciliste_info"]
        elif any(word in query_lower for word in ["sveučilišt", "jurja", "dobril", "pul"]):
            related_categories = ["sveuciliste_info", "kontakt", "studijski_programi"]
        elif any(word in query_lower for word in ["ECTS", "bod", "godina", "potvrda"]):
            related_categories = ["studiranje", "ispiti", "kontakt"]
        else:
            # Ako ne možemo odrediti kategoriju, koristimo općenite
            related_categories = ["sveuciliste_info", "kontakt", "studijski_programi"]
        
        suggested_questions = []
        
        # Iz svake relevantne kategorije uzmi po jedno pitanje ako postoji
        for category in related_categories:
            if self.categories[category] and len(self.categories[category]) > 0:
                # Ako kategorija ima pitanja, odaberi prvo koje nije trenutno pitanje
                for question in self.categories[category]:
                    if question != query and question not in suggested_questions:
                        suggested_questions.append(question)
                        break
        
        # Ako nismo uspjeli dobiti dovoljno pitanja, dodaj neka općenita
        if len(suggested_questions) < 3:
            general_questions = [
                "Koje je službeno ime Sveučilišta u Puli?",
                "Kako mogu kontaktirati studentsku službu?",
                "Gdje mogu pronaći raspored predavanja?",
                "Koji su rokovi za prijavu ispita?",
                "Koje fakultete ima Sveučilište Jurja Dobrile u Puli?"
            ]
            
            for question in general_questions:
                if question != query and question not in suggested_questions:
                    suggested_questions.append(question)
                    if len(suggested_questions) >= 3:
                        break
        
        # Vrati maksimalno 3 prijedloga
        return suggested_questions[:3]

    async def get_response(self, query: str) -> dict:
        """
        Dobij odgovor na temelju korisničkog upita
        
        Args:
            query (str): Korisničko pitanje
            
        Returns:
            dict: Rječnik koji sadrži tekst odgovora i prijedloge za sljedeća pitanja
        """
        print(f"Obrada upita: '{query}'")
        
        # Pozovi sinkronu verziju koja koristi manual threshold routing
        return self.get_response_sync(query)