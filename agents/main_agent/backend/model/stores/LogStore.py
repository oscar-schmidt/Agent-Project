from typing import Any, Optional, List
from pydantic import BaseModel, Field
from constants import SYSTEM_LOG_LIST


class LogStore(BaseModel):
    log_placeholder: Optional[Any] = None
    logs: List[Any] = Field(default_factory=list)
    system_log_list: List[Any] = SYSTEM_LOG_LIST

    model_config = {
        "arbitrary_types_allowed": True
    }

    def append(self, log: Any):
        self.logs.append(log)

    def extend(self, log_list: list[str]):
        for log in log_list:
            if self.logs.__contains__(log):
                self.logs.append(log)
