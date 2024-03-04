from .ICCEEndpoint import ICCEEndpoint
from ..utils import Status, INVALID_ID

import numpy as np
import time

class ICCEInterface:
    
    def __init__(self, frequency_hz=120, agent_hint = INVALID_ID, ip_addr = 'localhost'):
        # ICCE attributes
        self.n_observation: int
        self.n_action: int
        self.observation : np.ndarray
        self.reward: float
        self.term: bool
        self.trunc: bool
        self.info: dict

        # ICCE settings
        self.id = INVALID_ID
        self.status = Status.SUCCESS
        self.episode = 0
        self.frequency_seconds = 1.0 / frequency_hz
        self.agent_hint = agent_hint

        # Communication layer endpoint
        self._endpoint = ICCEEndpoint(ip_addr=ip_addr)

    def run(self):
        """ Runs the ICCE client.

        Refer to the Sequence Diagram for a clearer picture. The flow of the ICCE client is

        ICCE handshakes with the Environment, validating its input/output sizes, and assigns an ICCE ID
        Samples the Environment for initial environment data (Observation, Reward, Terminated, Truncated, Info)
        Loop while the Environment does not shut down (frequency-bound):
            Calls act() user-defined interface to take an action in the Simulation
            Samples the Environment for environment data after taking action
            Calls post_sample() user-defined interface to run behaviors after taking an action in the simulation
            If received end-of-episode, calls user-defined interface post_episode to run behaviours after the end of an episode

        Raises:
            NotImplementedError: If user-defined interfaces, act(), post_sample(), post_episode(), are not implemented by the interfacing ICCE.
        """
        # Handshake with environment and validate I/O of model with environment - Blocking
        self._handshake_and_validate()

        # Pull initial data
        self._sample()

        # Main loop
        while True:
            # sample at fixed interval
            start = time.perf_counter()

            match(self.status):
                case Status.SUCCESS:
                    # ICCE to act
                    self._act()

                    # Sample the environment after taking an action
                    self._sample()

                    # Post sample - Learn/Remember, depends on algorithm
                    self.post_sample(observation=self.observation, reward=self.reward)
                case Status.DONE:
                    print('end of episode...')
                    self.post_episode()
                    self.status = Status.WAIT # Wait for sample to retrieve status == SUCCESS
                case Status.WAIT:
                    # Continue sampling for status change to SUCCESS
                    self._sample()
                case Status.SHUTDOWN:
                    print('shutting down...')
                    exit(code=1)

            # Check interval
            # delay = desired_interval - delta_time
            delta = time.perf_counter() - start
            delay = self.frequency_seconds - delta
            if delay > 0: # positive delay -> faster than expected
                time.sleep(delay)

    # USER-DEFINED INTERFACES
    def act(self, observation: np.ndarray) -> np.ndarray:
        """ Sets the action of the Simulation agent.

        {USER-DEFINED} This is an interface which is used to set the self.action attribute in the ICCE.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing ICCE.
        """
        raise NotImplementedError("Functionality to infer actions must be defined!")

    def post_sample(self, observation: np.ndarray, reward: float):
        """ User-defined behaviour after sampling the Environment.

        This function is called after taking an action in the environment and sampling the environment after. Examples of use cases
        of this function is to allow the RL agent to learn using the new observations and reward after taking an action.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing ICCE.
        """
        raise NotImplementedError("Functionality to infer actions must be defined!")

    def post_episode(self):
        """ User-defined behaviour after an episode terminates/truncates.

        This function is called after receiving a signal from the Environment indicating that the episode has terminated/truncated.

        Raises:
            NotImplementedError: If this function is not implemented by the interfacing ICCE.
        """
        raise NotImplementedError("Functionality to infer actions must be defined!")
        
    # HELPERS
    def _handshake_and_validate(self) -> bool:
        print("Handshake and validate\n----------")
        print(f"{self.n_observations}  {self.n_actions}")

        # Invoke RPC
        response = self._endpoint.handshake_and_validate(self.n_observations, self.n_actions, self.agent_hint)

        ## Handshake failed
        if Status(response.status) != Status.SUCCESS:
            print('Handshake failed.')
            match(response.status):
                case Status.OBSERVATION_SIZE_ERROR:
                    print('n_observations do not match.')
                case Status.ACTION_SIZE_ERROR:
                    print('n_actions do not match.')
                case Status.ICCE_ID_ERROR:
                    print('Failed to generate ICCE ID. Maximum Agents mapped or invalid Simulation Agent ID hinted.')
                case Status.AGENT_ID_ERROR:
                    print('Failed to map Agent ID. Maximum Agents mapped.')
            # End program
            exit()
        
        ## Handshake success
        self.id = response.id
        print('Handshake successful. ICCE ID : ', self.id)

    def _sample(self):
        # Invoke RPC
        response = self._endpoint.sample(id=self.id)

        # Cache into memory
        self.observation = np.frombuffer(
            buffer=response.observation,
            dtype=np.float32
        )
        self.reward = response.reward
        self.term = response.terminated
        self.trunc = response.truncated
        self.episode = response.episode
        # If post_episode() executed (moved to WAIT), do not set status back to DONE
        status = Status(response.status)
        if status == Status.DONE and self.status == Status.WAIT:
            return
        self.status = status

    def _act(self):
        # Call user-defined act() which sets self.action
        action = self.act(self.observation)

        # Invoke RPC
        _ = self._endpoint.act(id=self.id, action=action.tobytes())


