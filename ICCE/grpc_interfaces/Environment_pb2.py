# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Environment.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11\x45nvironment.proto\x12\x0b\x45nvironment\"\x1e\n\x10HandshakeRequest\x12\n\n\x02id\x18\x01 \x01(\x05\"#\n\x11HandshakeResponse\x12\x0e\n\x06status\x18\x01 \x01(\x05\"\x1a\n\x0cStartRequest\x12\n\n\x02id\x18\x01 \x01(\x05\"\x1f\n\rStartResponse\x12\x0e\n\x06status\x18\x01 \x01(\x05\"\x1c\n\x0e\x45nvDataRequest\x12\n\n\x02id\x18\x01 \x01(\x05\"2\n\x0f\x45nvDataResponse\x12\x0e\n\x06status\x18\x01 \x01(\x05\x12\x0f\n\x07\x65pisode\x18\x02 \x01(\x05\"\x1b\n\rActionRequest\x12\n\n\x02id\x18\x01 \x01(\x05\" \n\x0e\x41\x63tionResponse\x12\x0e\n\x06status\x18\x01 \x01(\x05\x32\xd0\x02\n\x0b\x45nvironment\x12Y\n\x16handshake_and_validate\x12\x1d.Environment.HandshakeRequest\x1a\x1e.Environment.HandshakeResponse\"\x00\x12K\n\x10start_simulation\x12\x19.Environment.StartRequest\x1a\x1a.Environment.StartResponse\"\x00\x12K\n\x0cget_env_data\x12\x1b.Environment.EnvDataRequest\x1a\x1c.Environment.EnvDataResponse\"\x00\x12L\n\x0fset_action_data\x12\x1a.Environment.ActionRequest\x1a\x1b.Environment.ActionResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'Environment_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_HANDSHAKEREQUEST']._serialized_start=34
  _globals['_HANDSHAKEREQUEST']._serialized_end=64
  _globals['_HANDSHAKERESPONSE']._serialized_start=66
  _globals['_HANDSHAKERESPONSE']._serialized_end=101
  _globals['_STARTREQUEST']._serialized_start=103
  _globals['_STARTREQUEST']._serialized_end=129
  _globals['_STARTRESPONSE']._serialized_start=131
  _globals['_STARTRESPONSE']._serialized_end=162
  _globals['_ENVDATAREQUEST']._serialized_start=164
  _globals['_ENVDATAREQUEST']._serialized_end=192
  _globals['_ENVDATARESPONSE']._serialized_start=194
  _globals['_ENVDATARESPONSE']._serialized_end=244
  _globals['_ACTIONREQUEST']._serialized_start=246
  _globals['_ACTIONREQUEST']._serialized_end=273
  _globals['_ACTIONRESPONSE']._serialized_start=275
  _globals['_ACTIONRESPONSE']._serialized_end=307
  _globals['_ENVIRONMENT']._serialized_start=310
  _globals['_ENVIRONMENT']._serialized_end=646
# @@protoc_insertion_point(module_scope)
