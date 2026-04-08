from uuid import uuid4
from typing import Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataCleaningAction, DataCleaningObservation
except ImportError:
    from models import DataCleaningAction, DataCleaningObservation

# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "id": "easy",
        "description": (
            "Fix the date formats in the following rows. "
            "All dates must follow YYYY-MM-DD format. "
            "Return the same rows with only the 'dob' field corrected."
        ),
        "messy_data": [
            {"id": 1, "name": "Alice", "dob": "01/14/1995"},
            {"id": 2, "name": "Bob",   "dob": "23-07-1988"},
            {"id": 3, "name": "Carol", "dob": "March 5, 2000"},
        ],
        "expected": [
            {"id": 1, "name": "Alice", "dob": "1995-01-14"},
            {"id": 2, "name": "Bob",   "dob": "1988-07-23"},
            {"id": 3, "name": "Carol", "dob": "2000-03-05"},
        ],
    },
    "medium": {
        "id": "medium",
        "description": (
            "Fill in the missing values in the following rows. "
            "Use sensible defaults: unknown names → 'Unknown', "
            "missing ages → 0, missing emails → 'no-email@example.com'. "
            "Return all rows with every field filled."
        ),
        "messy_data": [
            {"id": 1, "name": None,    "age": 25,   "email": "alice@example.com"},
            {"id": 2, "name": "Bob",   "age": None, "email": "bob@example.com"},
            {"id": 3, "name": "Carol", "age": 30,   "email": None},
            {"id": 4, "name": None,    "age": None, "email": None},
        ],
        "expected": [
            {"id": 1, "name": "Unknown",              "age": 0,  "email": "alice@example.com"},
            {"id": 2, "name": "Bob",                  "age": 0,  "email": "bob@example.com"},
            {"id": 3, "name": "Carol",                "age": 30, "email": "no-email@example.com"},
            {"id": 4, "name": "Unknown",              "age": 0,  "email": "no-email@example.com"},
        ],
    },
    "hard": {
        "id": "hard",
        "description": (
            "This dataset has multiple issues: wrong date formats, missing values, "
            "and duplicate rows. Fix ALL issues: "
            "1) Dates must be YYYY-MM-DD. "
            "2) Missing names → 'Unknown', missing ages → 0, missing emails → 'no-email@example.com'. "
            "3) Remove duplicate rows (keep first occurrence). "
            "Return only unique, fully cleaned rows."
        ),
        "messy_data": [
            {"id": 1, "name": "Alice", "dob": "01/14/1995", "age": 29,   "email": "alice@example.com"},
            {"id": 2, "name": None,    "dob": "23-07-1988", "age": None, "email": "bob@example.com"},
            {"id": 3, "name": "Carol", "dob": "March 5, 2000", "age": 24, "email": None},
            {"id": 2, "name": None,    "dob": "23-07-1988", "age": None, "email": "bob@example.com"},  # duplicate
        ],
        "expected": [
            {"id": 1, "name": "Alice",   "dob": "1995-01-14", "age": 29, "email": "alice@example.com"},
            {"id": 2, "name": "Unknown", "dob": "1988-07-23", "age": 0,  "email": "bob@example.com"},
            {"id": 3, "name": "Carol",   "dob": "2000-03-05", "age": 24, "email": "no-email@example.com"},
        ],
    },
}

# ── Grader ────────────────────────────────────────────────────────────────────

def grade(submitted: list, expected: list) -> tuple[float, str]:
    """Score submitted rows vs expected. Returns (score 0.0-1.0, feedback)."""
    if not submitted:
        return 0.0, "No data submitted."

    # Remove extra rows beyond expected length
    submitted = submitted[: len(expected)]

    if len(submitted) != len(expected):
        return 0.1, f"Expected {len(expected)} rows, got {len(submitted)}."

    total_fields = 0
    correct_fields = 0
    feedback_lines = []

    for i, (sub_row, exp_row) in enumerate(zip(submitted, expected)):
        for key, exp_val in exp_row.items():
            total_fields += 1
            sub_val = sub_row.get(key)
            # Compare as strings, lowercased, stripped
            if str(sub_val).strip().lower() == str(exp_val).strip().lower():
                correct_fields += 1
            else:
                feedback_lines.append(
                    f"Row {i+1} field '{key}': expected '{exp_val}', got '{sub_val}'"
                )

    score = round(correct_fields / total_fields, 2) if total_fields > 0 else 0.0
    feedback = (
        "Perfect!" if score == 1.0
        else "Issues: " + "; ".join(feedback_lines[:3])  # show first 3 errors
    )
    return score, feedback


# ── Environment ───────────────────────────────────────────────────────────────

class DataCleaningEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[str] = None
        self._last_score: float = 0.0

    def reset(self, task_id: str = "easy") -> DataCleaningObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = task_id if task_id in TASKS else "easy"
        self._last_score = 0.0
        task = TASKS[self._current_task]

        return DataCleaningObservation(
            task_id=self._current_task,
            task_description=task["description"],
            messy_data=task["messy_data"],
            score=0.0,
            feedback="Task started. Clean the data and submit.",
            done=False,
            reward=0.0,
        )

    def step(self, action: DataCleaningAction) -> DataCleaningObservation:
        self._state.step_count += 1
        task = TASKS.get(self._current_task or "easy")

        score, feedback = grade(action.cleaned_data, task["expected"])
        self._last_score = score
        done = score >= 1.0 or self._state.step_count >= 3

        return DataCleaningObservation(
            task_id=self._current_task or "easy",
            task_description=task["description"],
            messy_data=task["messy_data"],
            score=score,
            feedback=feedback,
            done=done,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._state