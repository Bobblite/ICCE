from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class HandshakeRequest(_message.Message):
    __slots__ = ("n_observations", "n_actions")
    N_OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    N_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    n_observations: int
    n_actions: int
    def __init__(self, n_observations: _Optional[int] = ..., n_actions: _Optional[int] = ...) -> None: ...

class HandshakeResponse(_message.Message):
    __slots__ = ("id", "status")
    ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    id: int
    status: int
    def __init__(self, id: _Optional[int] = ..., status: _Optional[int] = ...) -> None: ...

class SampleRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class SampleResponse(_message.Message):
    __slots__ = ("observation", "reward", "terminated", "truncated", "episode", "status")
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    TERMINATED_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    observation: bytes
    reward: float
    terminated: bool
    truncated: bool
    episode: int
    status: int
    def __init__(self, observation: _Optional[bytes] = ..., reward: _Optional[float] = ..., terminated: bool = ..., truncated: bool = ..., episode: _Optional[int] = ..., status: _Optional[int] = ...) -> None: ...

class ActionRequest(_message.Message):
    __slots__ = ("id", "action")
    ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    id: int
    action: bytes
    def __init__(self, id: _Optional[int] = ..., action: _Optional[bytes] = ...) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: int
    def __init__(self, status: _Optional[int] = ...) -> None: ...
