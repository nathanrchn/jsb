from json import dumps
from typing import Any, Dict, Callable


Schema = Dict[str, Any]
FormatPrompt = Callable[[Schema], str]

DEFAULT_FORMAT_PROMPT: FormatPrompt = lambda schema: dumps(schema)
