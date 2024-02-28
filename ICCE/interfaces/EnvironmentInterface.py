from .EnvironmentEndpoint import EnvironmentEndpoint
from ..utils import Status, INVALID_ID

import numpy as np
import time
import threading

class EnvironmentInterface:
    def __init__(self, frequency_hz=240, max_episodes=10, time_between_episodes=3, debug = False):
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

        # Debug attributes
        self.debug = debug
        if self.debug:
            self.debug_n = 1
            self.cur_avg = 0.0

        # Maps between ICCE and Simulation Agents
        self._icce_to_sim_agent = {}
        self._sim_agent_to_icce = {}
        self.registered_agents = []

        # Communication layer endpoint
        self._endpoint = EnvironmentEndpoint(
            handshake_cb=self._on_handshake_and_validate,
            sample_cb=self._on_sample,
            act_cb=self._on_act
        )

        # Mutex lock
        self.lock = threading.Lock()
    
    @property
    def active_agents(self):
        return self._sim_agent_to_icce.keys()


    ########## KEY FUNCTIONALITIES ##########
    def run(self):
        """ Runs the environment.
        
        Refer to the Sequence Diagram for a clearer picture. The flow of the Environment is

        Environment resets the Simulation using the user-defined interface `reset()`\n
        {Ad-hoc} Environment validates ICCE input/output sizes, assigns ICCE IDs and maps Simulation Agent ID to ICCE ID\n
        Loop while episode count < max episodes (frequency-bound):\n
            Samples simulation data from Simulation -> Compute Observations, Rewards, Terminated, Truncated, Info of ALL ICCEs\n
            {Ad-hoc} Sets simulation agent actions in Simulation when RPC is invoked by ICCE clients\n
            if episode is terminated/truncated:\n
                sleep() to allow icce client to sample the end of episode\n
                resets the environment and simulation\n
        Shutdown the environment

        Raises:
            NotImplementedError: If any combination of interfaces, reset(), sample(), act(), is not implemented by user.
        """
        # Instantiate environment data
        self.observations = np.ndarray(shape=(len(self.registered_agents), self.n_observation), dtype=np.float64)
        self.actions = np.ndarray(shape=(len(self.registered_agents), self.n_action), dtype=np.float32)
        self.rewards = np.ndarray(shape=(len(self.registered_agents)), dtype=np.float64)
        self.term = np.ndarray(shape=(len(self.registered_agents)), dtype=bool)
        self.trunc = np.ndarray(shape=(len(self.registered_agents)), dtype=bool)
        self.info = [{} for _ in range(len(self.registered_agents))]

        # Reset simulation and pull initial data
        self._reset()

        # Start gRPC server
        self._endpoint.start()
        
        # Main update loop -> Until max_episodes hit
        self._update()
        
        # Set status to shutdown
        self.status = Status.SHUTDOWN
        # Sleep to allow time for ICCEs to sample this shutdown status
        print('Shutting down...')
        time.sleep(10)
        self._endpoint.shutdown()

    def register(self, agent_id):
        """ Registers a Simulation agent.

        Agents are to be registered before run() is called by passing an identifier which is recognized by the Simulations as an argument. By
        registering agents, the Environment will pull Simulation data and compute Environment data via the user-defined interface
        sample(). The order of registration denotes the ICCE ID of that agent.

        Args:
            agent_id : The identifier of the agent recognized by the Simulation.
        """
        self.registered_agents.append(agent_id)

    def _reset(self):
        """ Resets the Simulation and Environment

        Calls the user-defined interface reset() which resets the Simulation. Resets the environment and computes the
        initial environment data by sampling the Simulation.

        Raises:
            NotImplementedError: If user-defined interfaces reset() is not implemented.
        """
        # Print average delta
        if self.debug:
            self.debug_n = 1
            self.cur_avg = 0.0

        # Reset Simulation
        status = self.reset()

        if status:
            # Reset Environment
            self._sample_data()
            self.status = Status.SUCCESS

    def _sample_data(self):
        """ Samples and computes Environment data for all simulation agents.

        Samples the Simulation for simulation agent data and compute into Environment data
        (Observation, Reward, Terminated, Truncated, Info) and consolidates all agents' Environment
        data into one repository for ICCE clients to sample.

        Raises:
            NotImplementedError: If user-defined interface, sample(), is not implemeted by the user.
        """
        with self.lock:
            for icce_id in range(len(self.registered_agents)):
                self.observations[icce_id], self.rewards[icce_id], self.term[icce_id], self.trunc[icce_id], self.info[icce_id] = self.sample(self.registered_agents[icce_id])

    def _update(self):
        """ Main update loop.

        Main update loops which handles retrieving Environment Data from the Simulation at a sampling rate. Also
        handles end-of-episode (term/trunc) and sets the Environment's status code for ICCEs to sample.

        Raises:
            NotImplementedError: If user-defined interfaces, sample() and/or reset(), are not defined by the user.
        """
        # Sample from Simulation as long as episode count not reached
        while(self.episode < self.max_episodes):
            # sample at fixed interval
            start = time.perf_counter()

            # sample
            self._sample_data()

            # Check for end of episode
            if any(self.term) or any(self.trunc):
                if self.debug:
                    print(f'Average update delta time: {self.cur_avg} seconds')

                # Set status code to indicate term/trunc
                self.status = Status.DONE

                # Sleep to allow time for ICCEs to sample this truncated observation
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

            # Compute running average of delta
            if self.debug:
                self.cur_avg += (delta - self.cur_avg) / self.debug_n


    ########## USER-DEFINED INTERFACES ##########
    def reset(self):
        """ Interface to reset the simulation.
        
        {USER-DEFINED} This is an interface which is used to reset the Simulation to an initial state,
        ready for the next episode.

        Args:
            None

        Returns:
            A boolean indicating the success(True) or failure(False) of the reset.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing Environment.
        """
        raise NotImplementedError('Functionality to reset environment must be defined!')
    
    def sample(self, agent_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """ Interface to pull environment data from the Simulation.
        
        {MUST BE DEFINED} This is an interface function which retrieves simulation data for the specified Simulation Agent and converts it
        into environment data: observation, reward, term, trunc, info.

        Args:
            agent_id : The ID of the Simulation Agent to sample.

        Returns:
            observation, reward, term, trunc, info of the specified Sim Agent.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing Environment.
        """
        raise NotImplementedError('Functionality to retrieve environment data from Simulation must be defined!')
    
    def act(self, agent_id, action):
        """ Interface to set the action of an Agent in the Simulation.

        {MUST BE DEFINED} This is an interface function which sets the action of the specified Simulation Agent.

        Args:
            agent_id : The ID of the Simulation Agent to set the action of.
            action : The action of the Simulation as provided by the ICCE as an ndarray.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing Environment.
        """
        raise NotImplementedError('Functionality to set action for simulation agent must be defined!')
    

    ########## CORE CALLBACKS ##########
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
    
    def _on_sample(self, icce_id):
        """ Retrieves the environment data of a specified ICCE agent.

        Callback function to sample the environment data (Observation, Reward, Terminated, Truncated, Info) of a specified ICCE, 
        as well as the status of the environment (SUCCESS/DONE/WAIT). This funciton is called by the gRPC endpoint when the relevant
        RPC is invoked.

        Args:
            icce_id : The ID of the ICCE to sample.

        Returns:
            Observation (in bytes), Reward, Term, Trunc, Status of the ICCE.
        """
        return self.observations[icce_id].tobytes(), self.rewards[icce_id], self.term[icce_id], self.trunc[icce_id], self.info[icce_id], int(self.status)
    
    def _on_act(self, icce_id, action_bytes):
        """ Sets the action of a Simulation agent using the ICCE ID.

        Callback function to set the action of a Simulation agent. The action data is received from the ICCE client and the ICCE ID
        of the client is mapped to the Simulation Agent's ID. The action is then set by callin the user-defined interface act() to
        set the action of the agent in the Simulation.

        icce_id : The ID of the ICCE which is mapped to a Simulation Agent's ID.
        action_bytes : The requested action in bytes which is to be converted to an ndarray.

        Returns:
            Status of setting the action.
        """
        action = np.frombuffer(buffer=action_bytes, dtype=np.float32)
        status = self.act(agent_id=self._icce_to_sim_agent[icce_id], action=action)
        return status


    ########## HELPERS ##########
    def _generate_icce_id(self) -> int:
        """ Helper func to generate an ICCE ID.

        Generates an unused ICCE ID.

        Returns:
            The generated ICCE ID.
        """
        with self.lock:
            if len(self._icce_to_sim_agent) >= len(self.registered_agents):
                icce_id = INVALID_ID
            else:
                icce_id = (len(self._icce_to_sim_agent))

        return icce_id
    
    def _generate_agent_id(self) -> int:
        """ Helper func to retrieves an unused Agent ID.

        Returns an unused Agent ID that has not been mapped. The Agent IDs are retrieved from the list of
        registered agents.

        Returns:
            An unmapped Agent ID
        """
        agent_id = INVALID_ID
        with self.lock:
            for id in self.registered_agents:
                if id not in self.active_agents:
                    agent_id = id
                    break

        return agent_id
    
    def _add_icce(self, icce_id, agent_id):
        """ Helper func to map ICCE IDs to Agent IDs and vice-versa
        
            Args:
                icce_id : The ID of the ICCE.
                agent_id : The ID of the Simulation agent.
        """
        
        with self.lock:
            self._icce_to_sim_agent.update({icce_id:agent_id})
            self._sim_agent_to_icce.update({agent_id:icce_id})
    

