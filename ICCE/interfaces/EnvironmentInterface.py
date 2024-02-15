from .EnvironmentEndpoint import EnvironmentEndpoint

import numpy as np
import time
import threading

INVALID_ID = -1

class EnvironmentInterface:
    def __init__(self, frequency_hz=60, max_episodes=100, time_between_episodes=3):
        # Environment attributes
        self.n_observations: int # n_observation of an agent
        self.n_actions: int # n_action of an agent
        self.observations: np.ndarray
        self.rewards: np.ndarray
        self.term: np.ndarray
        self.trunc: np.ndarray
        self.info: dict()
        self.episode: int

        # Environment settings
        self.frequency_seconds = 1.0/frequency_hz
        self.max_episodes = max_episodes
        self.time_between_episodes = time_between_episodes

        # Maps between ICCE and Simulation Agents
        self._icce_to_sim_agent = dict()
        self._sim_agent_to_icce = dict()
        self._agents = []

        # Communication layer endpoint
        self._endpoint = EnvironmentEndpoint(
            handshake_cb=self._on_handshake_and_validate,
            env_data_cb=self._on_get_env_data
        )

        # Mutex lock
        self.lock = threading.Lock()
    
    @property
    def simulation_agents(self):
        return self._agents.copy()

    def run(self):
        """ Starts the environment. """

        # Reset simulation and pull initial data
        self.observations, self.info = self.reset()

        # Start gRPC server
        self._endpoint.start()

        # Sample from Simulation as long as episode count not reached
        while(self.episode < self.max_episodes):
            # sample at fixed interval
            self.observations, self.rewards, self.term, self.trunc, self.info = self._sample_data()

            # Check for end of episode
            if any(self.term) or any(self.trunc):
                # Sleep to allow time for ICCEs to sample this truncated observation
                time.sleep(self.time_between_episodes)

                # Reset Environment
                self.observations = np.full_like(self.observations, 0.0, dtype=self.observations.dtype)
                self.rewards = np.full_like(self.rewards, 0.0, dtype=self.rewards.dtype)
                self.term = np.full_like(self.term, False, dtype=self.term.dtype)
                self.trunc = np.full_like(self.trunc, False, dtype=self.trunc.dtype)

                # Reset Simulation and environment
                self.reset()

                # Increment episode count
                self.episode += 1

        self._endpoint.shutdown()

    def _interval(self, func):
        start = time.perf_counter()
        result = func()
        delay = self.frequency_seconds - (time.perf_counter() - start) # delay = desired_interval - delta_time
        if delay > 0: # positive delay -> faster than expected
            time.sleep(delay)
        return result
    
    @_interval
    def _sample_data(self):
        # TODO: Pull data for each Sim agent and organize into ndarrays based on ICCE IDs
        return self.sample()
    

    def add_agent(self, agent_id):
        self._agents.append(agent_id)
    
    # Interfaces
    def start(self) -> int:
        raise NotImplementedError('Functionality to start the simulation must be defined!')

    def reset(self) -> None:
        """ {MUST BE DEFINED} Interface to reset the simulation.
        
        This is an interface function which resets the simulation and returns observations and auxiliary information.

        Args:
            None

        Returns:
            Observations and auxiliary information for all agents.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing Environment.
        """
        raise NotImplementedError('Functionality to reset environment must be defined!')
    
    def sample(self, agent_id: int) -> (np.ndarray, float, bool, bool, dict):
        """ {MUST BE DEFINED} Interface to pull environment data from the simulation.
        
        This is an interface function which retrieves simulation data for the specified Simulation Agent and converts it
        into environment data: observation, reward, term, trunc, info.

        Args:
            agent_id : The ID of the Sim Agent to sample.

        Returns:
            observation, reward, term, trunc, info of the specified Sim Agent.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing Environment.
        """
        raise NotImplementedError('Functionality to retrieve environment data from Simulation must be defined!')
    
    def act(self, agent_id, action):
        raise NotImplementedError('Functionality to set action for simulation agent must be defined!')
    
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
        icce_id, status = self._generate_icce_id()
        if status != 1:
            return icce_id, status
        
        # Get an agent that has yet to be mapped
        agent_id = [id for id in self._agents if id not in self._sim_agent_to_icce.keys()][0]
        
        # Map icce to agent
        self._add_icce(icce_id, agent_id)

        return icce_id, 1

    # HELPERS
    def _generate_icce_id(self) -> (int, int):
        if len(self._icce_to_sim_agent) >= len(self._agents):
            return INVALID_ID, -3
        
        return (len(self._agents)), 1
    
    def _get_unmapped_sim_agent(self) -> (int, int):
        unmapped_agents = [id for id in self._agents if id not in self._sim_agent_to_icce.keys()]
        if not unmapped_agents:
            return INVALID_ID, -4
        
        return unmapped_agents[0], 1

    def _on_get_env_data(self, icce_id):
        # TODO: IMPLEMENT METHOD TO PULL ICCE DATA FROM ENV
        obs, reward, term, trunc, info = self.sample(icce_id)
        return obs.tobytes(), reward, term, trunc, info
    
    def _add_icce(self, icce_id, agent_id):
        self.lock.acquire()
        self._icce_to_sim_agent.update({icce_id:agent_id})
        self._sim_agent_to_icce.update({agent_id:icce_id})
        self.lock.release()