from enum import IntEnum

INVALID_ID = -1

class Status(IntEnum):
    SUCCESS = 1,
    DONE = 2,
    WAIT = 3,
    SHUTDOWN = 4,
    OBSERVATION_SIZE_ERROR = -1,
    ACTION_SIZE_ERROR = -2,
    ICCE_ID_ERROR = -3,
    AGENT_ID_ERROR = -4
