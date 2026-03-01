# MLLM Screenshot Vergleich

Ein Streamlit-Prototyp zum Vergleichen von Screenshot-Paaren über mehrere MLLMs (GPT-4V, Claude Vision, Gemini Pro).

## Features

- Screenshot-Paare aus Ordnern laden
- Konfigurierbare Namenskonventionen für Referenz-/Vergleichsbilder
- XML-strukturierte Prompts mit Template-System
- Parallele Analyse über 3 MLLMs (GPT-4V, Claude Vision, Gemini Pro)
- Fortschrittsanzeige während der Analyse
- Filterbare Ergebnisansicht
- CSV-Export mit konfigurierbaren Spalten

## Installation

```bash
# Virtual Environment erstellen
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt
```

## Konfiguration

### API Keys

Erstelle eine `.env` Datei basierend auf `.env.example`:

```bash
cp .env.example .env
```

Füge deine API Keys ein:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Screenshot-Ordnerstruktur

Die Screenshots müssen in folgender Struktur vorliegen:

```
data/screenshots/
├── paar_001/
│   ├── reference.png
│   └── comparison.png
├── paar_002/
│   ├── reference.png
│   └── comparison.png
└── ...
```

Die Dateinamen können angepasst werden (z.B. `_a`/`_b` oder `before`/`after`).

## Verwendung

```bash
streamlit run app.py
```

1. **Sidebar**: API Keys eingeben und Ordnerpfad konfigurieren
2. **Screenshots-Tab**: Ordner scannen und Paare anzeigen
3. **Analyse-Tab**: Element eingeben, MLLMs auswählen und Analyse starten
4. **Ergebnisse-Tab**: Ergebnisse filtern und als CSV exportieren

## Projektstruktur

```
Prototype/
├── app.py                    # Streamlit Hauptanwendung
├── requirements.txt          # Dependencies
├── config/
│   └── settings.py           # Konfiguration & API-Key Management
├── core/
│   ├── pair_loader.py        # Screenshot-Paar Erkennung
│   ├── prompt_builder.py     # XML-Prompt Konstruktion
│   └── result_manager.py     # Ergebnis-Aggregation & CSV-Export
├── providers/
│   ├── base.py               # Abstrakte Basisklasse
│   ├── openai_provider.py    # GPT-4V
│   ├── anthropic_provider.py # Claude Vision
│   └── google_provider.py    # Gemini Pro Vision
└── data/
    └── screenshots/          # Screenshot-Ordner
```

## Erweiterung

Um ein neues MLLM hinzuzufügen:

1. Erstelle einen neuen Provider in `providers/`
2. Erbe von `BaseMLLMProvider`
3. Implementiere die `analyze()`-Methode
4. Registriere den Provider in `app.py`

## Lizenz

MIT
