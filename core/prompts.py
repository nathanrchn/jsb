from json import dumps

from core.types import FormatPrompt


DEFAULT_FORMAT_PROMPT: FormatPrompt = (
    lambda schema: f" You need to generate a JSON object that matches the schema below.  Do not include the schema in the output and DIRECTLY return the JSON object without any additional information.  The schema is: {dumps(schema)}"
)
