from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


class SupervisorState(TypedDict):
    user_request: str
    agent_to_invoke: str
    agent_result: str
    conversation_history: Annotated[list, operator.add]
    is_complete: bool
