from enum import IntEnum

INVALID_ID = -1

class Status(IntEnum):
    SUCCESS = 1,
    OBSERVATION_SIZE_ERROR = -1,
    ACTION_SIZE_ERROR = -2,
    ICCE_ID_ERROR = -3,
    AGENT_ID_ERROR = -4,
    SHUTDOWN = -5
