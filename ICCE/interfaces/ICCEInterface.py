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
        while self.status == Status.SUCCESS:
            # sample at fixed interval
            start = time.perf_counter()
            self._sample()

            # DEBUG ONLY
            print(self.reward)

            # Check interval
            # delay = desired_interval - delta_time
            delta = time.perf_counter() - start
            delay = self.frequency_seconds - delta
            if delay > 0: # positive delay -> faster than expected
                time.sleep(delay)

        match(self.status):
            case Status.SHUTDOWN:
                print('shutting down...')

        
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
        self.status = Status(response.status)

