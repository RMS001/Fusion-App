from abc import ABC, abstractmethod


class Tool(ABC):
    """A built-in tool the model can call during the agentic loop.

    Descriptions are prompts too — write them imperatively and concretely,
    aimed at the model.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema for arguments

    #: Set False by subclasses whose backend/deps are missing or misconfigured;
    #: unavailable tools are never offered to the model.
    available: bool = True

    @abstractmethod
    async def execute(self, args: dict) -> str:
        """Return a plain-text result for the model.

        NEVER raise to the caller: catch internal errors and return a short
        'ERROR: ...' string instead, so the model can adapt and the request
        never dies.
        """
        ...

    def spec(self) -> dict:
        """OpenAI-style tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
