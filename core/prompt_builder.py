from dataclasses import dataclass
from typing import Optional


DEFAULT_TEMPLATE = """Du erhältst zwei Screenshots einer dynamischen Webseite.

- Screenshot A ist das Referenzbild.
- Screenshot B ist das Vergleichsbild.

Aufgabe:
1) Beschreibe jeden Screenshot kurz (1-2 Sätze), damit nachvollziehbar ist, was zu sehen ist.
2) Prüfe für jeden Screenshot separat, ob das Element "{element}" vorhanden ist.
3) Begründe kurz, warum du denkst, dass das Element vorhanden/nicht vorhanden ist.

Antworte als JSON-Objekt mit GENAU diesen Feldern:
{
  "description_a": string,
  "description_b": string,
  "presence_a": boolean,
  "presence_b": boolean,
  "reasoning": string
}

Wichtig:
- Gib nur JSON zurück (keine Erklärtexte, keine Markdown-Fences).
- reasoning soll sich sowohl auf Screenshot A als auch Screenshot B beziehen."""


STRUCTURED_TEMPLATE = """<prompt>
  <role>
    Du bist ein Experte für visuelle Qualitätssicherung (QA) und spezialisiert auf den Vergleich von Benutzeroberflächen (UI) in dynamischen Webanwendungen.
  </role>

  <context>
    Dir werden zwei Screenshots einer dynamischen Webseite zur Analyse vorgelegt:
    - Screenshot A dient als Referenzbild (Soll-Zustand oder Vergleichsbasis).
    - Screenshot B dient als Vergleichsbild (aktueller Zustand).
    Die Analyse findet im Rahmen eines automatisierten UI-Tests statt, bei dem die Konsistenz spezifischer Elemente geprüft wird.
  </context>

  <directive>
    Führe eine detaillierte visuelle Prüfung für das Element "{element}" durch.
    1. Analysiere beide Screenshots individuell.
    2. Stelle fest, ob das gesuchte Element in jedem der Bilder sichtbar und identifizierbar ist.
    3. Erstelle eine objektive Begründung für deine Entscheidung, die beide Bilder berücksichtigt.
  </directive>

  <style>
    - Antworte ausschließlich im JSON-Format.
    - Vermeide jegliche Einleitungstexte, Markdown-Code-Fences (wie ```json) oder abschließende Kommentare.
    - Die Sprache innerhalb der JSON-Werte muss Deutsch sein.
    - Halte die Beschreibungen prägnant und technisch sachlich.
  </style>

  <example>
    Input-Element: "Warenkorb-Button"
    Output:
    {
      "description_a": "Die Startseite mit blauem Header und einem deutlich sichtbaren Einkaufswagen-Icon oben rechts.",
      "description_b": "Die Startseite im mobilen Viewport; der Header ist komprimiert, das Icon fehlt.",
      "presence_a": true,
      "presence_b": false,
      "reasoning": "In Screenshot A ist das Icon mit der Beschriftung 'Warenkorb' klar im Header erkennbar. In Screenshot B wurde das Element vermutlich aufgrund des responsiven Designs in ein Burger-Menü verschoben und ist nicht direkt sichtbar."
    }
  </example>

  <output_format>
    {
      "description_a": "Kurze Beschreibung von Screenshot A (1-2 Sätze).",
      "description_b": "Kurze Beschreibung von Screenshot B (1-2 Sätze).",
      "presence_a": boolean,
      "presence_b": boolean,
      "reasoning": "Zusammenfassende Begründung für beide Ergebnisse."
    }
  </output_format>
</prompt>"""


@dataclass
class PromptTemplate:
    name: str
    template: str
    description: str = ""


class PromptBuilder:
    def __init__(self):
        self.templates: dict[str, PromptTemplate] = {
            "default": PromptTemplate(
                name="default",
                template=DEFAULT_TEMPLATE,
                description="Standard-Template für Element-Präsenz-Check"
            ),
            "structured": PromptTemplate(
                name="structured",
                template=STRUCTURED_TEMPLATE,
                description="XML-strukturiertes Template mit Rolle, Kontext und Beispiel"
            )
        }
        self.current_template = "default"
    
    def build(self, element: str, template_name: Optional[str] = None) -> str:
        """Build a prompt for the given element."""
        template_name = template_name or self.current_template
        template = self.templates.get(template_name, self.templates["default"])
        # Only replace the {element} placeholder. Using str.format() would treat
        # any JSON braces `{ ... }` in the template as formatting placeholders.
        return template.template.replace("{element}", element)
    
    def add_template(self, name: str, template: str, description: str = ""):
        """Add a new prompt template."""
        self.templates[name] = PromptTemplate(
            name=name,
            template=template,
            description=description
        )
    
    def set_current_template(self, name: str):
        """Set the default template to use."""
        if name in self.templates:
            self.current_template = name
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def list_templates(self) -> list[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def set_custom_template(self, template: str):
        """Set a custom template as the current one."""
        self.templates["custom"] = PromptTemplate(
            name="custom",
            template=template,
            description="Benutzerdefiniertes Template"
        )
        self.current_template = "custom"


def build_prompt(element: str, template: Optional[str] = None) -> str:
    """Convenience function to build a prompt."""
    builder = PromptBuilder()
    if template:
        builder.set_custom_template(template)
    return builder.build(element)
