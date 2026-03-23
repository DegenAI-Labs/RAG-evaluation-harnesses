import json
from typing import Any, Dict, List, Union


def doc_to_target(doc: Dict[str, Any]) -> Union[str, List[str]]:
    """PopQA gold labels: `possible_answers` may be a list or JSON string from TSV."""
    pa = doc["possible_answers"]
    if isinstance(pa, str):
        return json.loads(pa)
    return list(pa)
