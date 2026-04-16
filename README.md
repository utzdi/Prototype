# MLLM Screenshot-Vergleich – Prototyp

Ein Streamlit-basierter Prototyp zur automatisierten Analyse von Screenshot-Paaren mittels multimodaler Large Language Models (MLLMs). Entwickelt im Rahmen einer Bachelorarbeit am KIT.

---

## Voraussetzungen

- Python **3.10+**
- `pip` und `venv`
- API-Keys für mindestens einen der unterstützten Anbieter (OpenAI, Anthropic, Google)
- Optional: Zugang zum Remote-Ollama-Server (Institut)

---

## Installation & Setup

```bash
# 1. Repository klonen
git clone <repo-url>
cd Prototype

# 2. Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

# 3. Abhängigkeiten installieren
pip install -r requirements.txt

# 4. Umgebungsvariablen anlegen
cp .env.example .env
```

---

## Konfiguration

Öffne die `.env`-Datei und trage deine API-Keys ein:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Optional – Remote-Ollama (Institut)
OLLAMA_REMOTE_URL=https://ollama-webui.aifb-bis-gpu01.aifb.kit.edu/ollama
OLLAMA_REMOTE_API_KEY=...
```

Nicht benötigte Keys können leer bleiben – die entsprechenden Provider werden dann in der App deaktiviert.

---

## Anwendung starten

```bash
streamlit run app.py
```

Die App öffnet sich automatisch im Browser (`http://localhost:8501`).

**Workflow in der App:**

1. **Sidebar** – API-Keys eingeben und Ordnerpfad zu den Screenshots konfigurieren
2. **Screenshots-Tab** – Ordner scannen und Bild-Paare anzeigen
3. **Analyse-Tab** – UI-Element eingeben, MLLM(s) auswählen und Analyse starten
4. **Ergebnisse-Tab** – Ergebnisse filtern und als CSV exportieren

---

## Auswertung & Diagramme

Nach abgeschlossenen Analyse-Runs liegen die Ergebnisse als JSON-Dateien in `data/runs/` vor. Für die Auswertung stehen zwei Skripte zur Verfügung:

```bash
# Metriken berechnen (Accuracy, Precision/Recall/F1, Halluzinationen, Stabilität, Performance)
# Eingabe: data/runs/Final_Runs/   Ausgabe: evaluation_results/
python evaluate.py

# Diagramme für die Thesis generieren
# Ausgabe: evaluation_results/charts/
python generate_charts_651.py
```

---

## Repo-Struktur

```
Prototype/
│
├── app.py                          # Streamlit-Hauptanwendung (Einstiegspunkt)
├── evaluate.py                     # Auswertungsskript (Metriken aus Run-JSONs)
├── generate_charts_651.py          # Diagramm-Generierung für die Thesis
├── requirements.txt                # Python-Abhängigkeiten
├── .env.example                    # Vorlage für API-Keys
│
├── config/
│   └── settings.py                 # Globale Einstellungen & API-Key-Management
│
├── core/                           # Kern-Logik
│   ├── pair_loader.py              # Erkennung & Laden von Screenshot-Paaren
│   ├── prompt_builder.py           # Aufbau der XML-strukturierten Prompts
│   ├── result_manager.py           # Ergebnis-Aggregation & CSV-Export
│   └── run_store.py                # Persistierung der Analyse-Runs als JSON
│
├── providers/                      # MLLM-Anbindungen
│   ├── base.py                     # Abstrakte Basisklasse (BaseMLLMProvider)
│   ├── openai_provider.py          # GPT-4V / GPT-4o
│   ├── anthropic_provider.py       # Claude Vision
│   ├── google_provider.py          # Gemini Pro Vision
│   └── ollama_provider.py          # Lokale / Remote-Ollama-Modelle (z. B. LLaVA)
│
├── data/
│   ├── screenshots/                # Screenshot-Paare nach Kategorien geordnet
│   │   ├── fullset/                # Vollständiger Datensatz (alle Kategorien)
│   │   │   ├── paar_1-001/         # Kategorie 1 – Paar 001
│   │   │   │   ├── reference.png
│   │   │   │   ├── comparison.png
│   │   │   │   └── meta.json       # Metadaten (Ground Truth etc.)
│   │   │   ├── paar_2-001/ …       # Kategorie 2
│   │   │   ├── paar_3-001/ …       # Kategorie 3
│   │   │   └── paar_4-001/ …       # Kategorie 4
│   │   ├── screenshots_categorie_1/  # Kategorie 1 (separater Ordner für Runs)
│   │   ├── screenshots_categorie_2/
│   │   ├── screenshots_categorie_3/
│   │   └── screenshots_categorie_4/
│   │
│   └── runs/                       # Analyse-Ergebnisse (JSON pro Run)
│       ├── Final_Runs/             # Finale Runs für die Auswertung
│       └── run_<timestamp>.json    # Einzelne Analyse-Runs
│
└── evaluation_results/             # Ausgabe von evaluate.py & generate_charts_651.py
    ├── charts/                     # Diagramme als PNG
    │   ├── 1_accuracy_overall.png
    │   ├── 2_accuracy_by_category.png
    │   ├── 3_precision_recall_f1.png
    │   ├── 4_hallucination_rates.png
    │   ├── 5_stability_overall.png
    │   ├── 6_stability_by_category.png
    │   ├── 7_accuracy_heatmap.png
    │   └── 8_performance.png
    ├── accuracy_by_category.csv
    ├── hallucination_summary.csv
    ├── performance_metrics.csv
    ├── precision_recall.csv
    ├── stability.csv
    └── summary.csv                 # Gesamtübersicht aller Metriken
```

### Provider hinzufügen

1. Neue Datei in `providers/` anlegen
2. Von `BaseMLLMProvider` (`providers/base.py`) erben
3. Methode `analyze()` implementieren
4. Provider in `app.py` registrieren
