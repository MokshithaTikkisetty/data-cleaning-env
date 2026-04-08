from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional

class DataCleaningAction(Action):
    """Action - AI submits a cleaned version of the data."""
    cleaned_data: List[dict] = Field(..., description="List of cleaned rows as dictionaries")

class DataCleaningObservation(Observation):
    """Observation - what the AI sees each step."""
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="What the AI needs to do")
    messy_data: List[dict] = Field(default=[], description="The messy data to clean")
    score: float = Field(default=0.0, description="Score so far (0.0 to 1.0)")
    feedback: str = Field(default="", description="Feedback on last attempt")
    done: bool = Field(default=False, description="Is the task complete?")