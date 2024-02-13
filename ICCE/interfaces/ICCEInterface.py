from .ICCEEndpoint import ICCEEndpoint

import numpy as np

class ICCEInterface:
    
    def __init__(self):
        self.n_observations: int
        self.n_actions: int
        self.id = -1
        self._endpoint = ICCEEndpoint()

    def run(self):
        # Handshake with environment and validate I/O of model with environment - Blocking
        response = self._endpoint.handshake_and_validate(self.n_observations, self.n_actions)
        ## Handshake failed
        if response.status != 1:
            print('Handshake failed.')
            match(response.status):
                case -1:
                    print('n_observations do not match.')
                case -2:
                    print('n_actions do not match.')
            return
        ## Handshake success
        self.id = response.id
        print('Handshake successful. ICCE ID : ', self.id)

        # Request to start simulation - Blocking
        #response = self._endpoint.start_simulation(id=self.id)

        env_data = self._endpoint.get_env_data(id=1)
        obs = np.frombuffer(env_data.data.observations, dtype=np.float32)
        print(f'Received observations\n{obs}')
        action_resp = self._endpoint.set_action_data(id=1)