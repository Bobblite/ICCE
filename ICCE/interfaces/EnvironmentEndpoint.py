import grpc
from ..grpc_interfaces import Environment_pb2, Environment_pb2_grpc

import threading
from concurrent import futures

class EnvironmentServicer(Environment_pb2_grpc.EnvironmentServicer):
    def __init__(self, handshake_cb, sample_cb, act_cb):
        super().__init__()
        # Store callbacks
        self._on_handshake_and_validate = handshake_cb
        self._sample_cb = sample_cb
        self._act_cb = act_cb

    def handshake_and_validate(self, request, context):
        """ Servicer implementation of handshake_and_validate.
        
        This is an implementation of the gRPC servicer function `handshake_and_validate()`. Invoking this function invokes the
        callback function `_on_handshake_and_validate()` defined in the `EnvironmentInterface` class which validates the
        observation(RL model input) and action(RL model output) sizes against the derived environment.

        Args:
            request : The incoming gRPC request message, `HandshakeRequest` containing the ICCE's observation and action sizes.

        Returns:
            The gRPC response message `HandshakeResponse` containing the generated ICCE ID and validation status.

        Raises:
            None
        """
        id, status = self._on_handshake_and_validate(request.n_observations, request.n_actions, request.agent_hint)
        response = Environment_pb2.HandshakeResponse(id=id, status=int(status))
        return response
    
    def sample(self, request, context):
        # Sample environment data from simulation
        obs, reward, term, trunc, info, status = self._sample_cb(request.id)

        # Set response
        response = Environment_pb2.SampleResponse()
        response.observation = obs
        response.reward = reward
        response.terminated = term
        response.truncated = trunc
        # TODO: response.data.info
        response.episode = 1 # TODO
        response.status = status
        return response
        
    def act(self, request, context):
        self._act_cb(icce_id=request.id, action_bytes=request.action)
        response = Environment_pb2.ActionResponse(status=1)
        return response
    

class EnvironmentEndpoint():
    def __init__(self, handshake_cb, sample_cb, act_cb, ip_addr='localhost'):
        # gRPC server
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self._server_thread = threading.Thread(target=self.start_server)

        # Environment servicer
        self._servicer = EnvironmentServicer(handshake_cb=handshake_cb, sample_cb=sample_cb, act_cb=act_cb)

        Environment_pb2_grpc.add_EnvironmentServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(ip_addr+':50051')
    
    def start(self):
        """ Starts the Environment communication layer on a separate thread. """
        self._server_thread.start()

    def shutdown(self):
        """ Shutsdown the gRPC server and rejoin the communication layer thread. """
        self._server.stop(grace=None)
        self._server_thread.join()

    def start_server(self):
        """ Starts the gRPC server and blocks the calling thread until server shutsdown. """
        self._server.start()
        self._server.wait_for_termination()

