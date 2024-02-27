from .ICCEEndpoint import ICCEEndpoint
from ..utils import Status, INVALID_ID

import numpy as np
import time

class ICCEInterface:
    
    def __init__(self, frequency_hz=120):
        # ICCE attributes
        self.n_observation: int
        self.n_action: int
        self.observation : np.ndarray
        self.action: np.ndarray
        self.reward: float
        self.term: bool
        self.trunc: bool
        self.info: dict

        # ICCE settings
        self.id = INVALID_ID
        self.status = Status.SUCCESS
        self.episode = 0
        self.frequency_seconds = 1.0 / frequency_hz

        # Communication layer endpoint
        self._endpoint = ICCEEndpoint()

    def run(self):
        # Handshake with environment and validate I/O of model with environment - Blocking
        self._handshake_and_validate()

        # Main loop
        while True:
            # sample at fixed interval
            start = time.perf_counter()
            self._sample()

            match(self.status):
                case Status.SUCCESS:
                    # ICCE to act
                    self._act()
                    # Post sample - Learn/Remember, depends on algorithm
                    self.post_sample()
                case Status.DONE:
                    print('end of episode...')
                    self.post_episode()
                    self.status = Status.WAIT # Wait for sample to retrieve status == SUCCESS
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
    def act(self):
        """ Happens before sampling
        """
        raise NotImplementedError("Functionality to infer actions must be defined!")

    def post_sample(self):
        raise NotImplementedError("Functionality to infer actions must be defined!")

    def post_episode(self):
        raise NotImplementedError("Functionality to infer actions must be defined!")
        
    # HELPERS
    def _handshake_and_validate(self) -> bool:
        print("Handshake and validate\n----------")
        print(f"{self.n_observations}  {self.n_actions}")

        # Invoke RPC
        response = self._endpoint.handshake_and_validate(self.n_observations, self.n_actions)

        ## Handshake failed
        if Status(response.status) != Status.SUCCESS:
            print('Handshake failed.')
            match(response.status):
                case Status.OBSERVATION_SIZE_ERROR:
                    print('n_observations do not match.')
                case Status.ACTION_SIZE_ERROR:
                    print('n_actions do not match.')
                case Status.ICCE_ID_ERROR:
                    print('Failed to generate ICCE ID. Maximum Agents mapped.')
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
        self.act()

        # Invoke RPC
        _ = self._endpoint.act(id=self.id, action=self.action.tobytes())


