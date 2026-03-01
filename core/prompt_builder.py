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
