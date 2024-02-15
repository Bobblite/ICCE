import grpc
from ..grpc_interfaces import Environment_pb2, Environment_pb2_grpc

class ICCEEndpoint():
    def __init__(self):
        self._channel = grpc.insecure_channel('localhost:50051')
        self._stub = Environment_pb2_grpc.EnvironmentStub(self._channel)

    def handshake_and_validate(self, n_observations, n_actions):
        handshake_req = Environment_pb2.HandshakeRequest(
            n_observations=n_observations,
            n_actions=n_actions)
        
        return self._stub.handshake_and_validate(handshake_req)

    def get_env_data(self, id: int):
        icce_data = Environment_pb2.EnvDataRequest(id=id)
        return self._stub.get_env_data(icce_data)
    
    def set_action_data(self, id: int):
        action_req = Environment_pb2.ActionRequest(id=id)
        return self._stub.set_action_data(action_req)