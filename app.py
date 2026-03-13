import streamlit as st
from pathlib import Path
from datetime import datetime
import time

from config.settings import Settings
from core.pair_loader import PairLoader, ScreenshotPair
from core.prompt_builder import PromptBuilder, DEFAULT_TEMPLATE
from core.result_manager import ResultManager, ExportConfig, generate_export_filename
from providers import OpenAIProvider, OPENAI_VISION_MODELS, AnthropicProvider, ANTHROPIC_VISION_MODELS, GoogleProvider, GOOGLE_VISION_MODELS, OllamaProvider, check_ollama_status, AnalysisResult, DEFAULT_MAX_DIMENSION, DEFAULT_JPEG_QUALITY

st.set_page_config(
    page_title="MLLM Screenshot Vergleich",
    page_icon="🔍",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables."""
    if "settings" not in st.session_state:
        st.session_state.settings = Settings()
    if "pair_loader" not in st.session_state:
        st.session_state.pair_loader = PairLoader()
    if "prompt_builder" not in st.session_state:
        st.session_state.prompt_builder = PromptBuilder()
    if "result_manager" not in st.session_state:
        st.session_state.result_manager = ResultManager()
    if "pairs" not in st.session_state:
        st.session_state.pairs = []
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "llava:7b"
    if "ollama_enabled" not in st.session_state:
        st.session_state.ollama_enabled = False
    if "ollama_remote_model" not in st.session_state:
        st.session_state.ollama_remote_model = None
    if "ollama_remote_enabled" not in st.session_state:
        st.session_state.ollama_remote_enabled = False
    if "ollama_remote_vision_models" not in st.session_state:
        st.session_state.ollama_remote_vision_models = []
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "image_compression" not in st.session_state:
        st.session_state.image_compression = True
    if "max_image_dimension" not in st.session_state:
        st.session_state.max_image_dimension = DEFAULT_MAX_DIMENSION
    if "jpeg_quality" not in st.session_state:
        st.session_state.jpeg_quality = DEFAULT_JPEG_QUALITY
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-5.2"
    if "openai_detail" not in st.session_state:
        st.session_state.openai_detail = "auto"
    if "anthropic_model" not in st.session_state:
        st.session_state.anthropic_model = "claude-sonnet-4-6"
    if "google_model" not in st.session_state:
        st.session_state.google_model = "gemini-2.5-pro"


def render_sidebar():
    """Render the sidebar with configuration options."""
    st.sidebar.title("Konfiguration")
    
    st.sidebar.subheader("API Keys")
    
    openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state.settings.openai_api_key or "",
        type="password",
        help="Für GPT-4V"
    )
    anthropic_key = st.sidebar.text_input(
        "Anthropic API Key",
        value=st.session_state.settings.anthropic_api_key or "",
        type="password",
        help="Für Claude Vision"
    )
    google_key = st.sidebar.text_input(
        "Google API Key",
        value=st.session_state.settings.google_api_key or "",
        type="password",
        help="Für Gemini Pro"
    )
    
    if openai_key != st.session_state.settings.openai_api_key:
        st.session_state.settings.openai_api_key = openai_key
    if anthropic_key != st.session_state.settings.anthropic_api_key:
        st.session_state.settings.anthropic_api_key = anthropic_key
    if google_key != st.session_state.settings.google_api_key:
        st.session_state.settings.google_api_key = google_key
    
    if st.session_state.settings.has_openai():
        st.sidebar.subheader("OpenAI Einstellungen")
        st.session_state.openai_model = st.sidebar.selectbox(
            "OpenAI Modell",
            options=OPENAI_VISION_MODELS,
            index=OPENAI_VISION_MODELS.index(st.session_state.openai_model) if st.session_state.openai_model in OPENAI_VISION_MODELS else 0,
            help="Vision-Modell für die Bildanalyse"
        )
        st.session_state.openai_detail = st.sidebar.selectbox(
            "Bilddetail",
            options=["auto", "low", "high"],
            index=["auto", "low", "high"].index(st.session_state.openai_detail),
            help="low=schneller/günstiger, high=bessere Qualität, auto=automatisch"
        )

    if st.session_state.settings.has_anthropic():
        st.sidebar.subheader("Anthropic Einstellungen")
        st.session_state.anthropic_model = st.sidebar.selectbox(
            "Claude Modell",
            options=ANTHROPIC_VISION_MODELS,
            index=ANTHROPIC_VISION_MODELS.index(st.session_state.anthropic_model) if st.session_state.anthropic_model in ANTHROPIC_VISION_MODELS else 0,
            help="claude-sonnet-4-6 = beste Balance, opus = leistungsstärker, haiku = schnellster"
        )

    if st.session_state.settings.has_google():
        st.sidebar.subheader("Google Einstellungen")
        st.session_state.google_model = st.sidebar.selectbox(
            "Gemini Modell",
            options=GOOGLE_VISION_MODELS,
            index=GOOGLE_VISION_MODELS.index(st.session_state.google_model) if st.session_state.google_model in GOOGLE_VISION_MODELS else 0,
            help="gemini-2.5-pro = stabil & leistungsstark, gemini-3.1-pro-preview = neuestes Modell (Preview)"
        )

    st.sidebar.divider()
    
    st.sidebar.subheader("Screenshot-Ordner")
    folder_path = st.sidebar.text_input(
        "Ordnerpfad",
        value=str(st.session_state.settings.screenshot_folder),
        help="Pfad zum Ordner mit Screenshot-Paaren"
    )
    st.session_state.settings.screenshot_folder = Path(folder_path)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ref_pattern = st.text_input(
            "Referenz-Pattern",
            value=st.session_state.settings.reference_pattern,
            help="Dateinamen-Pattern für Referenzbilder"
        )
        st.session_state.settings.reference_pattern = ref_pattern
        st.session_state.pair_loader.reference_pattern = ref_pattern.lower()
    
    with col2:
        comp_pattern = st.text_input(
            "Vergleichs-Pattern",
            value=st.session_state.settings.comparison_pattern,
            help="Dateinamen-Pattern für Vergleichsbilder"
        )
        st.session_state.settings.comparison_pattern = comp_pattern
        st.session_state.pair_loader.comparison_pattern = comp_pattern.lower()
    
    if st.sidebar.button("Ordner scannen", use_container_width=True):
        scan_folder()
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Ollama (Lokal)")
    ollama_status = check_ollama_status()
    
    if ollama_status["running"]:
        st.sidebar.success("Ollama läuft")
        vision_models = ollama_status.get("vision_models", [])
        if vision_models:
            selected_model = st.sidebar.selectbox(
                "Vision Model",
                options=vision_models,
                index=0 if vision_models else 0,
                help="Lokales Vision-Modell auswählen"
            )
            st.session_state.ollama_model = selected_model
            st.session_state.ollama_enabled = True
        else:
            st.sidebar.warning("Keine Vision-Modelle gefunden. Installiere mit: `ollama pull llava:7b`")
            st.session_state.ollama_enabled = False
    else:
        st.sidebar.warning("Ollama nicht erreichbar")
        st.sidebar.caption("Starte Ollama mit: `ollama serve`")
        st.session_state.ollama_enabled = False
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Ollama (Remote/Institut)")
    settings = st.session_state.settings
    
    if settings.has_ollama_remote():
        remote_status = check_ollama_status(
            base_url=settings.ollama_remote_url,
            api_key=settings.ollama_remote_api_key
        )
        
        if remote_status["running"]:
            st.sidebar.success("Institut-Server verbunden")
            remote_vision = remote_status.get("vision_models", [])
            all_remote = remote_status.get("models", [])
            
            display_models = remote_vision if remote_vision else all_remote
            st.session_state.ollama_remote_vision_models = display_models
            
            if display_models:
                selected_remote = st.sidebar.selectbox(
                    "Remote Vision Model",
                    options=display_models,
                    index=0,
                    help="Vision-Modell auf dem Institut-Server"
                )
                st.session_state.ollama_remote_model = selected_remote
                st.session_state.ollama_remote_enabled = True
                
                if not remote_vision:
                    st.sidebar.caption("Keine bekannten Vision-Modelle erkannt. Alle Modelle werden angezeigt.")
            else:
                st.sidebar.warning("Keine Modelle auf dem Server gefunden")
                st.session_state.ollama_remote_enabled = False
        else:
            error_msg = remote_status.get("error", "Unbekannter Fehler")
            st.sidebar.error(f"Nicht erreichbar: {error_msg}")
            st.session_state.ollama_remote_enabled = False
    else:
        st.sidebar.info("Konfiguriere OLLAMA_REMOTE_URL und OLLAMA_REMOTE_API_KEY in .env")
        st.session_state.ollama_remote_enabled = False
    
    st.sidebar.divider()
    
    available = []
    if st.session_state.settings.has_openai():
        available.append(f"OpenAI ({st.session_state.openai_model})")
    if st.session_state.settings.has_anthropic():
        available.append(f"Anthropic ({st.session_state.anthropic_model})")
    if st.session_state.settings.has_google():
        available.append(f"Google ({st.session_state.google_model})")
    if st.session_state.ollama_enabled:
        available.append("LLaVA (Lokal)")
    if st.session_state.ollama_remote_enabled:
        available.append(f"{st.session_state.ollama_remote_model} (Remote)")
    
    if available:
        st.sidebar.success(f"Verfügbar: {', '.join(available)}")
    else:
        st.sidebar.warning("Keine MLLMs verfügbar")
    
    st.sidebar.divider()
    
    st.sidebar.subheader("Bildkomprimierung")
    st.session_state.image_compression = st.sidebar.checkbox(
        "Bilder vor dem Senden komprimieren",
        value=st.session_state.image_compression,
        help="Reduziert die Bildgröße um 413-Fehler bei Remote-Servern zu vermeiden"
    )
    if st.session_state.image_compression:
        st.session_state.max_image_dimension = st.sidebar.slider(
            "Max. Bildgröße (px)",
            min_value=512,
            max_value=2048,
            value=st.session_state.max_image_dimension,
            step=128,
            help="Längste Seite wird auf diesen Wert skaliert"
        )
        st.session_state.jpeg_quality = st.sidebar.slider(
            "JPEG Qualität",
            min_value=30,
            max_value=95,
            value=st.session_state.jpeg_quality,
            step=5,
            help="Höher = bessere Qualität, größere Datei"
        )
    
    st.sidebar.divider()
    st.session_state.debug_mode = st.sidebar.checkbox(
        "Debug-Modus",
        value=st.session_state.debug_mode,
        help="Zeigt detaillierte Informationen zu Anfragen und Antworten"
    )


def scan_folder():
    """Scan the configured folder for screenshot pairs."""
    folder = st.session_state.settings.screenshot_folder
    if folder.exists():
        pairs = st.session_state.pair_loader.scan_folder(folder)
        st.session_state.pairs = pairs
        st.toast(f"{len(pairs)} Paare gefunden")
    else:
        st.error(f"Ordner existiert nicht: {folder}")


def render_config_tab():
    """Render the configuration tab."""
    st.header("Prompt-Konfiguration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Prompt Template")
        
        template_options = st.session_state.prompt_builder.list_templates()
        selected_template = st.selectbox(
            "Template auswählen",
            options=template_options,
            index=template_options.index(st.session_state.prompt_builder.current_template)
        )
        st.session_state.prompt_builder.set_current_template(selected_template)
        
        current = st.session_state.prompt_builder.get_template(selected_template)
        custom_template = st.text_area(
            "Template bearbeiten",
            value=current.template if current else DEFAULT_TEMPLATE,
            height=400,
            help="Verwende {element} als Platzhalter für das zu prüfende Element"
        )
        
        if st.button("Als Custom speichern"):
            st.session_state.prompt_builder.set_custom_template(custom_template)
            st.success("Template gespeichert")
    
    with col2:
        st.subheader("Export-Optionen")
        
        config = st.session_state.result_manager.export_config
        
        config.include_descriptions = st.checkbox(
            "Bild-Beschreibungen einschließen",
            value=getattr(config, "include_descriptions", True),
            help="Speichert description_a/description_b in Ergebnissen und CSV"
        )
        config.include_reasoning = st.checkbox(
            "Begründung einschließen",
            value=config.include_reasoning
        )
        config.include_tokens = st.checkbox(
            "Token-Verbrauch einschließen",
            value=config.include_tokens
        )
        config.include_latency = st.checkbox(
            "Latenz einschließen",
            value=config.include_latency
        )
        config.include_raw_response = st.checkbox(
            "Raw Response einschließen",
            value=config.include_raw_response
        )
        config.include_timestamp = st.checkbox(
            "Zeitstempel einschließen",
            value=config.include_timestamp
        )
        
        st.divider()
        
        st.subheader("Vorschau")
        element_preview = st.text_input("Element für Vorschau", value="Impressum")
        if st.button("Prompt-Vorschau"):
            preview = st.session_state.prompt_builder.build(element_preview)
            st.code(preview, language="json")


def render_screenshots_tab():
    """Render the screenshots gallery tab."""
    st.header("Screenshot-Paare")
    
    pairs = st.session_state.pairs
    
    if not pairs:
        st.info("Keine Screenshot-Paare geladen. Bitte Ordner in der Sidebar scannen.")
        return
    
    st.success(f"{len(pairs)} Paare gefunden")
    
    cols_per_row = st.slider("Spalten", min_value=1, max_value=4, value=2)
    
    for i in range(0, len(pairs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(pairs):
                pair = pairs[i + j]
                with col:
                    render_pair_preview(pair)


def render_pair_preview(pair: ScreenshotPair):
    """Render a preview of a single screenshot pair."""
    with st.container(border=True):
        st.subheader(pair.pair_id)
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Referenz")
            if pair.reference_path.exists():
                st.image(str(pair.reference_path), use_container_width=True)
            else:
                st.error("Datei nicht gefunden")
        
        with col2:
            st.caption("Vergleich")
            if pair.comparison_path.exists():
                st.image(str(pair.comparison_path), use_container_width=True)
            else:
                st.error("Datei nicht gefunden")


def render_analysis_tab():
    """Render the analysis runner tab."""
    st.header("Analyse starten")
    
    pairs = st.session_state.pairs
    settings = st.session_state.settings
    
    if not pairs:
        st.warning("Keine Screenshot-Paare geladen.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Einstellungen")
        
        element = st.text_input(
            "Zu prüfendes Element",
            value="Impressum",
            help="Das Element, das in den Screenshots gesucht werden soll"
        )
        
        available_mllms = []
        if settings.has_openai():
            available_mllms.append(f"OpenAI ({st.session_state.openai_model})")
        if settings.has_anthropic():
            available_mllms.append(f"Anthropic ({st.session_state.anthropic_model})")
        if settings.has_google():
            available_mllms.append(f"Google ({st.session_state.google_model})")
        if st.session_state.ollama_enabled:
            available_mllms.append("LLaVA (Lokal)")
        if st.session_state.ollama_remote_enabled:
            available_mllms.append(f"{st.session_state.ollama_remote_model} (Remote)")
        
        if not available_mllms:
            st.error("Keine MLLMs verfügbar. Bitte API Keys konfigurieren oder Ollama starten.")
            return
        
        selected_mllms = st.multiselect(
            "MLLMs auswählen",
            options=available_mllms,
            default=available_mllms
        )
        
        pair_selection = st.radio(
            "Paare",
            options=["Alle", "Auswahl"],
            horizontal=True
        )
        
        if pair_selection == "Auswahl":
            selected_pairs = st.multiselect(
                "Paare auswählen",
                options=[p.pair_id for p in pairs],
                default=[p.pair_id for p in pairs[:5]]
            )
            pairs_to_analyze = [p for p in pairs if p.pair_id in selected_pairs]
        else:
            pairs_to_analyze = pairs
    
    with col2:
        st.subheader("Übersicht")
        st.metric("Paare", len(pairs_to_analyze))
        st.metric("MLLMs", len(selected_mllms))
        st.metric("Gesamt API-Calls", len(pairs_to_analyze) * len(selected_mllms))
    
    st.divider()
    
    if st.button("Analyse starten", type="primary", use_container_width=True, disabled=not selected_mllms):
        run_analysis(pairs_to_analyze, selected_mllms, element)


def run_analysis(pairs: list[ScreenshotPair], mllms: list[str], element: str):
    """Run the analysis for all pairs and MLLMs."""
    st.session_state.analysis_running = True
    st.session_state.result_manager.clear()
    
    settings = st.session_state.settings
    prompt = st.session_state.prompt_builder.build(element)
    
    providers = {}
    img_max = st.session_state.max_image_dimension if st.session_state.image_compression else 0
    img_quality = st.session_state.jpeg_quality if st.session_state.image_compression else 0
    
    openai_label = f"OpenAI ({st.session_state.openai_model})"
    if openai_label in mllms and settings.has_openai():
        providers[openai_label] = OpenAIProvider(
            api_key=settings.openai_api_key,
            model=st.session_state.openai_model,
            max_image_dimension=img_max,
            jpeg_quality=img_quality,
            image_detail=st.session_state.openai_detail,
        )
    anthropic_label = f"Anthropic ({st.session_state.anthropic_model})"
    if anthropic_label in mllms and settings.has_anthropic():
        providers[anthropic_label] = AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=st.session_state.anthropic_model,
            max_image_dimension=img_max,
            jpeg_quality=img_quality,
        )
    google_label = f"Google ({st.session_state.google_model})"
    if google_label in mllms and settings.has_google():
        providers[google_label] = GoogleProvider(
            api_key=settings.google_api_key,
            model=st.session_state.google_model,
            max_image_dimension=img_max,
            jpeg_quality=img_quality,
        )
    
    if "LLaVA (Lokal)" in mllms and st.session_state.ollama_enabled:
        providers["LLaVA (Lokal)"] = OllamaProvider(
            model=st.session_state.ollama_model,
            display_name="LLaVA (Lokal)",
            max_image_dimension=img_max,
            jpeg_quality=img_quality
        )
    
    remote_label = f"{st.session_state.ollama_remote_model} (Remote)"
    if remote_label in mllms and st.session_state.ollama_remote_enabled:
        providers[remote_label] = OllamaProvider(
            model=st.session_state.ollama_remote_model,
            base_url=settings.ollama_remote_url,
            api_key=settings.ollama_remote_api_key,
            display_name=remote_label,
            max_image_dimension=img_max,
            jpeg_quality=img_quality
        )
    
    st.session_state.debug_logs = []
    
    total_calls = len(pairs) * len(providers)
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    current_call = 0
    
    for pair in pairs:
        for mllm_name, provider in providers.items():
            current_call += 1
            status_text.text(f"Analysiere {pair.pair_id} mit {mllm_name}... ({current_call}/{total_calls})")
            
            result = provider.analyze(
                image_a_path=pair.reference_path,
                image_b_path=pair.comparison_path,
                prompt=prompt,
                pair_id=pair.pair_id,
                element=element
            )
            
            st.session_state.result_manager.add_result(result)
            if result.debug:
                st.session_state.debug_logs.append(result)
            
            progress_bar.progress(current_call / total_calls)
            
            with results_container:
                display_result_inline(result)
    
    status_text.text("Analyse abgeschlossen!")
    st.session_state.analysis_running = False
    st.balloons()


def display_result_inline(result: AnalysisResult):
    """Display a single result inline during analysis."""
    if result.error:
        st.error(f"{result.pair_id} | {result.mllm}: Fehler - {result.error}")
    else:
        match_icon = "✅" if result.match is True else ("❌" if result.match is False else "❓")
        presence_a = "Ja" if result.presence_a is True else ("Nein" if result.presence_a is False else "?")
        presence_b = "Ja" if result.presence_b is True else ("Nein" if result.presence_b is False else "?")
        st.write(f"{match_icon} {result.pair_id} | {result.mllm}: A={presence_a}, B={presence_b}")


def render_results_tab():
    """Render the results tab."""
    st.header("Ergebnisse")
    
    results = st.session_state.result_manager.get_results()
    
    if not results:
        st.info("Noch keine Ergebnisse vorhanden. Führe zuerst eine Analyse durch.")
        return
    
    summary = st.session_state.result_manager.get_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gesamt", summary["total"])
    col2.metric("Übereinstimmungen", summary["matches"])
    col3.metric("Unterschiede", summary["mismatches"])
    col4.metric("Fehler", summary["errors"])
    
    st.divider()
    
    st.subheader("Nach MLLM")
    for mllm, stats in summary["by_mllm"].items():
        with st.expander(f"{mllm} ({stats['total']} Ergebnisse)"):
            cols = st.columns(3)
            cols[0].metric("Total", stats["total"])
            cols[1].metric("Matches", stats["matches"])
            cols[2].metric("Fehler", stats["errors"])
    
    st.divider()
    
    st.subheader("Detaillierte Ergebnisse")
    
    df = st.session_state.result_manager.to_dataframe()
    
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        mllm_filter = st.multiselect(
            "MLLM Filter",
            options=df["mllm"].unique().tolist(),
            default=df["mllm"].unique().tolist()
        )
    with filter_col2:
        match_filter = st.multiselect(
            "Match Filter",
            options=["True", "False", "None"],
            default=["True", "False", "None"]
        )
    
    filtered_df = df[df["mllm"].isin(mllm_filter)]
    
    match_values = []
    if "True" in match_filter:
        match_values.append(True)
    if "False" in match_filter:
        match_values.append(False)
    if "None" in match_filter:
        match_values.append(None)
    
    filtered_df = filtered_df[filtered_df["match"].isin(match_values) | filtered_df["match"].isna()]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    st.divider()
    
    st.subheader("Export")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        filename = st.text_input(
            "Dateiname",
            value=generate_export_filename("results")
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("CSV herunterladen", type="primary"):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )


def render_debug_tab():
    """Render the debug tab with detailed request/response information."""
    st.header("Debug-Informationen")
    
    if not st.session_state.debug_mode:
        st.info("Debug-Modus ist deaktiviert. Aktiviere ihn in der Sidebar.")
        return
    
    debug_results = st.session_state.debug_logs
    
    if not debug_results:
        st.info("Noch keine Debug-Daten vorhanden. Führe zuerst eine Analyse durch.")
        return
    
    st.success(f"{len(debug_results)} Anfragen protokolliert")
    
    for i, result in enumerate(debug_results):
        debug = result.debug
        if not debug:
            continue
        
        status = "Fehler" if result.error else "OK"
        header = f"{result.pair_id} | {result.mllm} [{status}]"
        
        with st.expander(header, expanded=(result.error is not None)):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Anfrage")
                st.markdown(f"**Endpoint:** `{debug.api_endpoint}`")
                st.markdown(f"**Modell:** `{debug.model_used}`")
                st.markdown(f"**Payload-Größe:** {debug.request_payload_size_kb:.0f} KB")
                
                st.markdown("---")
                st.markdown("**Bild A (Referenz)**")
                st.markdown(
                    f"- Original: {debug.image_a_resolution_original}, {debug.image_a_size_original_kb:.0f} KB\n"
                    f"- Gesendet: {debug.image_a_resolution_sent}, {debug.image_a_size_sent_kb:.0f} KB"
                )
                
                st.markdown("**Bild B (Vergleich)**")
                st.markdown(
                    f"- Original: {debug.image_b_resolution_original}, {debug.image_b_size_original_kb:.0f} KB\n"
                    f"- Gesendet: {debug.image_b_resolution_sent}, {debug.image_b_size_sent_kb:.0f} KB"
                )
                
                total_saved = (
                    debug.image_a_size_original_kb + debug.image_b_size_original_kb -
                    debug.image_a_size_sent_kb - debug.image_b_size_sent_kb
                )
                if total_saved > 0:
                    st.markdown(f"**Ersparnis:** {total_saved:.0f} KB eingespart")
            
            with col2:
                st.subheader("Antwort")
                if result.error:
                    st.error(f"**Fehler:** {result.error}")
                else:
                    st.markdown(f"**Latenz:** {result.latency_ms:.0f} ms")
                    if result.tokens_used:
                        st.markdown(f"**Tokens:** {result.tokens_used}")
                    st.markdown(f"**Presence A:** {'Ja' if result.presence_a is True else ('Nein' if result.presence_a is False else '?')}")
                    st.markdown(f"**Presence B:** {'Ja' if result.presence_b is True else ('Nein' if result.presence_b is False else '?')}")
                    st.markdown(f"**Match:** {'Ja' if result.match is True else ('Nein' if result.match is False else '?')}")
                    if result.description_a:
                        st.markdown(f"**Description A:** {result.description_a}")
                    if result.description_b:
                        st.markdown(f"**Description B:** {result.description_b}")
                    if result.reasoning:
                        st.markdown(f"**Reasoning:** {result.reasoning}")
            
            with st.expander("Gesendeter Prompt"):
                st.code(debug.prompt_sent, language="json")
            
            if result.raw_response:
                with st.expander("Raw Response"):
                    st.code(result.raw_response)


def main():
    """Main application entry point."""
    init_session_state()
    
    st.title("MLLM Screenshot Vergleich")
    st.caption("Vergleiche Screenshot-Paare mit mehreren MLLMs")
    
    render_sidebar()
    
    tabs = ["Konfiguration", "Screenshots", "Analyse", "Ergebnisse"]
    if st.session_state.debug_mode:
        tabs.append("Debug")
    
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:
        render_config_tab()
    
    with tab_objects[1]:
        render_screenshots_tab()
    
    with tab_objects[2]:
        render_analysis_tab()
    
    with tab_objects[3]:
        render_results_tab()
    
    if st.session_state.debug_mode and len(tab_objects) > 4:
        with tab_objects[4]:
            render_debug_tab()


if __name__ == "__main__":
    main()
