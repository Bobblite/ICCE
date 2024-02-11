#!/bin/bash

mkdir ./ICCE/grpc_interfaces -p
python -m grpc_tools.protoc -I./ICCE/protos --python_out=./ICCE/grpc_interfaces --pyi_out=./ICCE/grpc_interfaces --grpc_python_out=./ICCE/grpc_interfaces ./ICCE/protos/*
sed --i '5s/^/from . /' ./ICCE/grpc_interfaces/*_pb2_grpc.py