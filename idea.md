# [2026-03-05] RAG for learning

---

# Dictionary

- [TC] - To consider
- [PTD] - Potentially to do

# The idea:

Pomysł zakłada stworzenie narzędzia, które byłoby pomocne w procesie nauki. Narzędzie powinno być oparte na mechanizmach wykorzystywanych przy RAGach oraz innych ciekawych konceptach, które są aktualnie trendy w dziedzinie AI.

# The 2 main features:

- Nauka (mniej ważny na początek) - Narzędzie powinno wspierać proces uczenia się, przyswajania wiedzy oraz przerabiania materiału.
    - **Czemu?** - Zauważyłem, że tracę sporo czasu na pisanie promptów, kiedy się uczę, a z reguły nauka odbywa się w bardzo podobny sposób.
    - **Zamysł?** - Wyobrażam sobie, że ten system mógłby zostać zorganizowany jak klasyczne drzewo plików, tzn. tworzymy kursy/przedmioty jako foldery, a następnie podfoldery do studiowania mniejszych część np. tematów. Powinniśmy stworzyć możliwość dodawania plików w różnych formatach, które następnie przechodziłyby przez: 
    Pierwszy etap RAG, czyli indexing. Zwróciłbym tutaj uwagę szczególnie na pliki .pdf, do których IBM stworzył parser Docling, można spróbować z opcją do_ocr, żeby odczytać tekst z obrazków, potencjalnie można użyć Presidio które jest PII (znajduje prywatne dane w tekście, których nie chcemy storować w vector db). 
    [PTD] W przyszłości można dodać model image-to-text (vision LLM e.g. LLaVA via Ollama), który dodowałby również informacje z infografik do textu [/PTD]. [TC] Jak zrobić chunking - czy osobno przetwarzamy każdą stronę, czy należy je jakoś łączyć? Page-based or Structure based [/TC] 
    [TC] 
    Embeddings - options to consider: sentence_transformers (same model for indexing and retrieval!) from HuggingFace Hub, or Ollama embeddings - could be matching the way LLMs are used. Also cloud, paid options available like openai one  [/TC]
    [TC] Vector store
    ChromaDB - embedded in Python process, stored locally
    Qdrant - run as separate Docker container, better for production
    Others: PineCone, Weaviate -> both stored on cloud, Weaviate also Docker available, more complex; FAISS no persistence, no metadata storage [/TC]
    A następnie agent AI powinien nas uczyć. Możemy dodać opcję, że piszemy w jaki sposób ma nas uczyć, ale domyślnie proponuje, żeby agent przeanalizował cały materiał i stwierdził, czego wgl chcemy się uczyć. A następnie powinien w prosty sposób wytłumaczyć ogólny zarys tematu. [PTD] Mi podobałby się feature polegający na tym, że gdzieś w systemie zapisany jest plik, który określa nasze umiejętności z poszczególnych dziedzin, który po sesji nauki/powtórki byłby modyfikowany przez LLM, a w kolejnych sesjach nauki byłby on odczytywany i dodawany do prompta, żeby agent określił na jakim poziomie jesteśmy i na jakich konceptach może bazować -> BAZA DANYCH [/PTD]. Oprócz tego agent może sam zaplanować sesję nauki i możemy się uczyć tylko na podstawie konwersacji z nim, albo możemy stworzyć narzędzie, które pozwoli na wybieranie konkretnych slajdów/fragmentów z pliku z prośbą o wytłumaczenie. Oczywiście za każdym razem odpowiedni kontekst powinien został dodany w ramach RAGa. System powinien dostawać feedback w sposób niewymagający dużo wysiłku od użytkownika i przeanalizować, czy powinien wytłumaczyć to zjawisko w prosty/intuicyjny sposób albo nauczyć użytkownika jakiegoś innego konceptu, na którym bazuje zrozumienie tego konkretnego. Jednakże, najważniejsze dla mnie byłoby, żeby po takiej sesji nauki użytkownik mógł dodać daną prezentację/materiał do powtórek. --> gdzie to zapisujemy (POSTGRESql może)
     Wtedy system powinien na podstawie konwersacji zapisać w odpowiednim pliku złożoność materiału, stopień opanowania oraz stworzyć zestaw zagadnień wraz z zakresem, które powinny zostać powtórzone. Powinien zwrócić również uwagę na fakt z czym mieliśmy największy problem. Oprócz tego system powinien zaktualizować plik z tym co dany użytkownik potrafi.
- Powtórki - Narzędzie powinno w sposób inteligentny planować powtórki w taki sposób, żeby zoptymalizować liczbę materiału, która zostaje w głowie w środowisku ograniczonego czasu pracy.
    - **Czemu?** - Prywatnie posiadam taki problem, że powtórki zwykle zabierają sporo czasu, a po drugie wymagają różnych ilości czasu w różne dni. Dodatkowo zwykłe czytanie materiału nie daje za dużo, więc i tak muszę wkleić materiał do jakiegoś LLMa, który przygotowuje test. Z tego też powodu narzędzie wyobrażam sobie w następujący sposób: powinien zostać zaimplementowany algorytm SM-2 lub FSRS, które znane są [m.in](http://m.in). z ANKI. Algorytmy te optymalizują powtórki w czasie zgodnie z krzywymi zapominanie, żeby jak najwięcej wiedzy zostało w głowie przy minimalnej ilości powtórek. Taki algorytm obliczałby, który materiał powinien zostać powtórzony danego dnia, a agent po zebraniu tego całego materiału oraz zdobyciu informacji od użytkownika [TC] być może za pomocą jakiejś synergii z kalendarzem Google [/TC] powinien rozplanować proces powtórki w taki sposób, żeby użytkownik był +/- w stanie powtórzyć materiał w optymalny sposób biorąc pod uwagę ograniczony czas. Agent powinien układać zadania/pytania i inne ćwiczenia sprawdzające na podstawie danych o trudności materiału, ograniczeniach czasowych, ilości materiału. Agent powinien w szczególności uwzględniać obszary, w których użytkownik popełniał błędy poprzednim razem oraz zapisywać takie ważne informacje w odpowiedniej bazie danych, która byłaby pomocna przy kolejnych powtórkach. Oczywiście poza samymi zadaniami powinno znaleźć się również okienko, w którym będziemy w stanie zadać pytanie do danego tematu/agent będzie nas naprowadzał na poprawne rozwiązanie. W tym miejscu też przyda się RAG, żeby informacje były jak najbardziej zbliżone do tych z materiału.

# (Subjectively) Nice features to implement:

- Stworzenie mechnizmu, który dostanie plik z informacjami o użytkowniku/sam na podstawie konwersacji scharakteryzuje użytkownika, a następnie będzie wspierał proces nauki odwołując się do tych informacji.
    - **Dlaczego?** - Mózg przyswaja lepiej informacje jeżeli są one jakkolwiek nacechowane emocjonalnie. Wydaje mi się, że lepiej zapamiętałbym dane słówko/koncept jeżeli odnosiłoby się do dziedziny, którą się interesuje/ulubionej postaci z filmu/bajki. Wydaje mi się, że może to być szczególnie przydatne przy zachęceniu małych dzieci do nauki. Fajnym pomysłem mogą też w takiej sytuacji być jakieś nagrody. Pamiętam, że w przedszkolu dostawaliśmy znaczki pokroju “mistrz ortografi” i mega zachęcało to do nauki xD
- Gamification - dodanie klasycznych elementów znanych z gier takich jak motywujące streaki, ranking ze znojmymi
- System AI, który analizowałby co znajduje się w danym folderze/pliku i tworzył krótkie podsumowania skracające czas przeszukiwania. Mogłoby to zostać rozszerzone o system rekomendacyjny, który na tej podstawie sprawdzałby czy mamy jakieś materiały na dany temat / w ramach którego przedmiotu zajmowaliśmy się danym zagadnieniem.
- Można dodać AI do automatycznego tworzenia notatek lub konwertowania podsumowania materiału na audio, żeby można było uczyć się w tramwaju / na siłowni.
- Forma zadań w powtórkach powinna być różnorodna. Z tego względu może można pobawić się z wieloma modelami, które będą wyspecjalizowane w tworzeniu danej formy zadań. Fajne mogłyby też być pytania na które trzeba udzielić odpowiedź otwartą a system sprawdzałby czy jest ona poprawna. Takie coś może być szczególnie przydatne przy nauce słówek i tłumaczeniu ich na polski, kiedy użyjemy synonimu, bo systemy oparte na fiszkach jak Quizlet często nie zaliczają wtedy odpowiedzi.
- Fajnie byłoby podabawić się w optymalizację kosztową i proste zapytania wysyłać do jakiegoś małego modelu, który zostałby odpalony lokalnie po kwantyzacji, a trudniejsze zadania leciałyby do jakiegoś w miarę flagowego LLMa po API.
- Mixture of Experts - można poszukać jakiegoś modelu typu mixture of experts, w ramach którego będą znajdowały się małe sieci, które są ekspertami w danej dziedzinie, co pozwoli znacząco ograniczyć koszty
- Narzędzie to generowania pytań / odpowiadania na pytania z bazy w oparciu o slajdy z konkretnego przedmiotu
# Technologies
Docker vs venv/uv
Backend: FastAPI
Frontend: React, Streamlit
Possibly adding Next.js api layer for auth, non-ai operations

vector search vs BM25
possibly reranking retrieved chunks

LLM: 
locally: Ollama
models can be downloaded from HuggingFace Hub
cloud api: Groq - 14k requests/day free tier
GoogleAI studio 1.5k req/day
HuggingFace Inference API rate-limited

fastapi.responses -> StreamingResponse

database storage of conversation - PostgreSQL

LiteLLM - one interface for many LLM providers

pipe: LangChain - for agents
LLamaIndex - specifically made for RAG use cases