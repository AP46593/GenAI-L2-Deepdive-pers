from typing import TypedDict, List, Optional

class RequiredState(TypedDict):
    messages: List[dict]
    uploaded_files: List[str]
    task_intent: str

class OptionalState(TypedDict, total=False):
    transcript: Optional[str]
    doc_summary: Optional[str]
    redacted_file: Optional[str]
    ocr_output: Optional[str]
    vector_store_path: Optional[str]

class AppState(RequiredState, OptionalState):
    pass