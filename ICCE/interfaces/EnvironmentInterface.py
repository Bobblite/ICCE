from .EnvironmentEndpoint import EnvironmentEndpoint

import numpy as np
import time
import threading

INVALID_ID = -1

class EnvironmentInterface:
    def __init__(self, expected_agents: int|None = None):
        # Interface attributes
        self.expected_agents = expected_agents
        self.is_running = False

        # Environment attributes
        self.n_observations: int # n_observation of an agent
        self.n_actions: int # n_action of an agent

        # Maps between ICCE and Simulation Agents
        self._icce_to_sim_agent = dict()
        self._sim_agent_to_icce = dict()

        # Communication layer endpoint
        self._endpoint = EnvironmentEndpoint(
            handshake_cb=self._on_handshake_and_validate,
            env_data_cb=self._on_get_env_data
        )

        # Mutex lock
        self.lock = threading.Lock()
    
    def run(self):
        """ Starts the environment. """
        self._endpoint.start()
        time.sleep(5)
        self._endpoint.shutdown()

    # Interfaces
    def generate_id(self) -> (int, int):
        """ {MUST BE DEFINED} Interface to generate and map ICCE ID to Sim Agent's ID.
        
        This is an interface function which generates an ICCE ID and maps it to an 
        agent in your simulation.

        Args:
            None

        Returns:
            The generated ICCE ID and the Simulation Agent ID.

        Raises:
            NotImplementedError: If this function is not implemented by the derived Environment.
        """
        raise NotImplementedError('Functionality to generate ICCE ID must be defined!')
    
    def sample(self, icce_id: int) -> (np.ndarray, float, bool, bool, dict):
        """ {MUST BE DEFINED} Interface to pull environment data from the simulation.
        
        This is an interface function which retrieves simulation data for the specified ICCE and converts it
        into environment data: observation, reward, term, trunc, info.

        Args:
            icce_id : The ID of the ICCE to sample.

        Returns:
            observation, reward, term, trunc, info of the specified ICCE.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing Environment.
        """
        raise NotImplementedError('Functionality to retrieve environment data from Simulation must be defined!')
    
    def reset(self) -> None:
        """ {MUST BE DEFINED} Interface to reset the simulation.
        
        This is an interface function which resets the simulation.

        Args:
            None

        Returns:
            None

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing Environment.
        """
        raise NotImplementedError('Functionality to reset environment must be defined!')

    # Core callbacks
    def _on_handshake_and_validate(self, n_observations: int, n_actions: int) -> (int, int):
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
        # I/O validation
        if (self.n_observations != n_observations):
            return INVALID_ID, -1
        if (self.n_actions != n_actions):
            return INVALID_ID, -2
        
        # Generate ICCE ID using interfaced function
        icce_id, agent_id = self.generate_id()

        # Type check
        if not isinstance(icce_id, int):
            raise TypeError('Implementation of `generate_id` must return an integer!')
        
        # Add to map
        self.add_icce(icce_id, agent_id)

        return icce_id, 1

    def _on_get_env_data(self, icce_id):
        obs, reward, term, trunc, info = self.sample(icce_id)
        return obs.tobytes(), reward, term, trunc, info

    # HELPER FUNCTIONS
    def add_icce(self, icce_id: int, agent_id: int):
        # Ensure only one thread has access to map at a time
        self.lock.acquire()
        self._icce_to_sim_agent.update({icce_id:agent_id})
        self._icce_to_sim_agent.update({agent_id:icce_id})
        self.lock.release()
    
    def remove_icce(self, icce_id: int):
        # Ensure only one thread has access to map at a time
        self.lock.acquire()
        del self._sim_agent_to_icce[self._icce_to_sim_agent[icce_id]]
        del self._icce_to_sim_agent[icce_id]
        self.lock.release()