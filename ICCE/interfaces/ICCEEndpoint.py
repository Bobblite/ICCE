import grpc
from ..grpc_interfaces import Environment_pb2, Environment_pb2_grpc

class ICCEEndpoint():
    
    def __init__(self):
        self._channel = grpc.insecure_channel('localhost:50051')
        self._stub = Environment_pb2_grpc.EnvironmentStub(self._channel)

    def get_env_data(self, id: int):
        icce_data = Environment_pb2.ICCE(id=id)
        return self._stub.GetEnvData(icce_data)