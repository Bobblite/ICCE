from .EnvironmentEndpoint import EnvironmentEndpoint

import time

INVALID_ID = -1

class EnvironmentInterface:
    def __init__(self):
        self.n_observations: int
        self.n_actions: int

        self._endpoint = EnvironmentEndpoint(handshake_cb=self._on_handshake_and_validate)
        self._icce_to_sim_agent = dict()
        self._sim_agent_to_icce = dict()
    
    def run(self):
        """ Starts the environment. """
        self._endpoint.start()
        time.sleep(5)
        self._endpoint.shutdown()

    # User-defined callbacks
    def generate_id(self):
        """ {MUST BE DEFINED} Interface to generate and map ICCE ID to Sim Agent's ID.
        
        This is an interface function which generates an ICCE ID and maps it to an 
        agent in your simulation. Use `add_icce()` after generating an ICCE ID to map the ICCE ID to an agent id.

        Args:
            None

        Returns:
            The generated ICCE ID.

        Raises:
            NotImplementedError: If this function is not implemented by the derived Environment.
        """
        raise NotImplementedError('Functionality to generate ICCE ID must be defined!')

    # Core callbacks
    def _on_handshake_and_validate(self, n_observations, n_actions):
        """ Callback function used to generate ICCE ID and validate observation/action spaces.
        
        This is function is called when the RPC method `handshake_and_validate` is invoked. An ICCE ID is generated using the implemented
        interface method `generate_id()` defined when interfacing the `EnvironmentInterface` class. The observation/action sizes of the model
        in the ICCE are also validated against that of the interfaced environment.

        Status code:
            -1 : Invalid observation size
            -2 : Invalid action size
            1: Success

        Args:
            n_observations : The observation size, or the input of the ICCE.
            n_actions: The action size, or the output of the ICCE.

        Returns:
            The generated ICCE ID based on the defined interface and the validation status.

        Raises:
            None
        """

        if (self.n_observations != n_observations):
            return INVALID_ID, -1
        if (self.n_actions != n_actions):
            return INVALID_ID, -2
        
        return self.generate_id(), 1
        

    # HELPER FUNCTIONS
    def add_icce(self, icce_id: int, agent_id: int):
        self._icce_to_sim_agent.update({icce_id:agent_id})
        self._icce_to_sim_agent.update({agent_id:icce_id})
    
    def remove_icce(self, icce_id: int):
        del self._sim_agent_to_icce[self._icce_to_sim_agent[icce_id]]
        del self._icce_to_sim_agent[icce_id]