from .EnvironmentEndpoint import EnvironmentEndpoint
from ..utils import Status, INVALID_ID

import numpy as np
import time
import threading

class EnvironmentInterface:
    def __init__(self, frequency_hz=240, max_episodes=3, time_between_episodes=3):
        # Environment attributes
        self.n_observation: int # n_observation of an agent
        self.n_action: int # n_action of an agent
        self.observations: np.ndarray
        self.actions: np.ndarray
        self.rewards: np.ndarray
        self.term: np.ndarray
        self.trunc: np.ndarray
        self.info: list

        # Environment settings
        self.episode = 0
        self.status = Status.SUCCESS
        self.frequency_seconds = 1.0/frequency_hz
        self.max_episodes = max_episodes
        self.time_between_episodes = time_between_episodes

        # Maps between ICCE and Simulation Agents
        self._icce_to_sim_agent = {}
        self._sim_agent_to_icce = {}
        self.registered_agents = []

        # Communication layer endpoint
        self._endpoint = EnvironmentEndpoint(
            handshake_cb=self._on_handshake_and_validate,
            sample_cb=self._on_sample
        )

        # Mutex lock
        self.lock = threading.Lock()
    
    @property
    def active_agents(self):
        return self._sim_agent_to_icce.keys()

    def run(self):
        """ Starts the environment. """
        # Instantiate environment data
        self.observations = np.ndarray(shape=(len(self.registered_agents), self.n_observation), dtype=np.float32)
        self.actions = np.ndarray(shape=(len(self.registered_agents), self.n_action), dtype=np.float32)
        self.rewards = np.ndarray(shape=(len(self.registered_agents)), dtype=np.float32)
        self.term = np.ndarray(shape=(len(self.registered_agents)), dtype=bool)
        self.trunc = np.ndarray(shape=(len(self.registered_agents)), dtype=bool)
        self.info = [{} for _ in range(len(self.registered_agents))]

        # Reset simulation and pull initial data
        self._reset()

        # Start gRPC server
        self._endpoint.start()
        
        # Sample from Simulation as long as episode count not reached
        while(self.episode < self.max_episodes):
            # sample at fixed interval
            start = time.perf_counter()
                
            # sample
            self._sample_data()

            # Check for end of episode
            if any(self.term) or any(self.trunc):
                print("terminated")
                # Sleep to allow time for ICCEs to sample this truncated observation
                # TODO: Or maybe use status code to indicate term/trunc and wait for all ICCEs to acknoledge
                # TODO-RE: Maybe not, as we want simulation to keep running regardless of number of ICCEs
                time.sleep(self.time_between_episodes)

                # Reset Simulation and environment
                self._reset()

                # Increment episode count
                self.episode += 1

            # Check interval
            # delay = desired_interval - delta_time
            delta = time.perf_counter() - start
            delay = self.frequency_seconds - delta
            if delay > 0: # positive delay -> faster than expected
                time.sleep(delay)

        # Set status to shutdown
        self.status = Status.SHUTDOWN
        # TODO: Maybe wanna use Status code to indicate shutting down and wait for all ICCE to acknoledge?
        # TODO RE: Maybe not, as we want simulation to keep running regardless of number of ICCEs
        # TODO RE2: Anyways when ICCE invokes RPC but server doesnt respond, they get an error anyways
        # Sleep to allow time for ICCEs to sample this shutdown status
        print('Shutting down...')
        time.sleep(10)
        self._endpoint.shutdown()

    def _reset(self):
        # Reset Simulation
        self.reset()
        print('reset')
        # Reset Environment
        self._sample_data()

    def _sample_data(self):
        self.lock.acquire()
        for icce_id in range(len(self.registered_agents)):
            self.observations[icce_id], self.rewards[icce_id], self.term[icce_id], self.trunc[icce_id], self.info[icce_id] = self.sample(self.registered_agents[icce_id])
        self.lock.release()

    def register_agent(self, agent_id):
        self.registered_agents.append(agent_id)
    
    # Interfaces
    def start(self) -> int:
        raise NotImplementedError('Functionality to start the simulation must be defined!')

    def reset(self):
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
    
    def sample(self, agent_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
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
    def _on_handshake_and_validate(self, n_observation: int, n_actions: int) -> tuple[int, Status]:
        """ Callback function used to generate ICCE ID and validate observation/action spaces.
        
        This is function is called when the RPC method `handshake_and_validate` is invoked. An ICCE ID is generated using the implemented
        interface method `generate_id()` defined when interfacing the `EnvironmentInterface` class. The observation/action sizes of the model
        in the ICCE are also validated against that of the interfaced environment.

        Args:
            n_observation : The observation size, or the input of the ICCE.
            n_actions: The action size, or the output of the ICCE.

        Returns:
            The generated ICCE ID based on the defined interface and the validation status.

        Raises:
            None
        """
        # I/O validation
        if (self.n_observation != n_observation):
            return INVALID_ID, Status.OBSERVATION_SIZE_ERROR
        if (self.n_action != n_actions):
            return INVALID_ID, Status.ACTION_SIZE_ERROR
        # Generate ICCE ID using interfaced function
        icce_id = self._generate_icce_id()
        if icce_id == INVALID_ID:
            return INVALID_ID, Status.ICCE_ID_ERROR
        
        # Get an agent that has yet to be mapped
        agent_id = self._generate_agent_id()
        if agent_id == INVALID_ID:
            return INVALID_ID, Status.AGENT_ID_ERROR
        
        # Map icce to agent
        self._add_icce(icce_id, agent_id)

        # Print log
        print("New ICCE registered. ICCE : Agent map")
        for agent in self._sim_agent_to_icce.keys():
            print(f"{agent} : {self._sim_agent_to_icce[agent]}")

        return icce_id, Status.SUCCESS

    # HELPERS
    def _generate_icce_id(self) -> int:
        self.lock.acquire()
        if len(self._icce_to_sim_agent) >= len(self.registered_agents):
            icce_id = INVALID_ID
        else:
            icce_id = (len(self._icce_to_sim_agent))
        self.lock.release()
        return icce_id
    
    def _generate_agent_id(self) -> int:
        agent_id = INVALID_ID
        self.lock.acquire()
        for id in self.registered_agents:
            if id not in self.active_agents:
                agent_id = id
                break
        self.lock.release()
        return agent_id

    def _on_sample(self, icce_id):
        return self.observations[icce_id].tobytes(), self.rewards[icce_id], self.term[icce_id], self.trunc[icce_id], self.info[icce_id], int(self.status)
    
    def _add_icce(self, icce_id, agent_id):
        self.lock.acquire()
        self._icce_to_sim_agent.update({icce_id:agent_id})
        self._sim_agent_to_icce.update({agent_id:icce_id})
        self.lock.release()

