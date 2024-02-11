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

class StartRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class StartResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: int
    def __init__(self, status: _Optional[int] = ...) -> None: ...

class EnvDataRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class EnvDataResponse(_message.Message):
    __slots__ = ("status", "episode")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    status: int
    episode: int
    def __init__(self, status: _Optional[int] = ..., episode: _Optional[int] = ...) -> None: ...

class ActionRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class ActionResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: int
    def __init__(self, status: _Optional[int] = ...) -> None: ...
