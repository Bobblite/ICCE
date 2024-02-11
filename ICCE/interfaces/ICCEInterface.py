from .ICCEEndpoint import ICCEEndpoint

class ICCEInterface:
    
    def __init__(self):
        self.n_observations: int
        self.n_actions: int
        
        self._endpoint = ICCEEndpoint()

    def run(self):
        # Connect to environment and validate I/O of model with environment
        response = self._endpoint.handshake_and_validate(self.n_observations, self.n_actions)
        # Handshake failed
        if response.status != 1:
            print('Handshake failed.')
            match(response.status):
                case -1:
                    print('n_observations do not match.')
                case -2:
                    print('n_actions do not match.')
            return
        # Handshake success
        print('Handshake successful. ICCE ID : ', response.id)

        start_resp = self._endpoint.start_simulation(id=1)
        env_data = self._endpoint.get_env_data(id=1)
        print(f'Received status: {env_data.status} | episode: {env_data.episode}')
        action_resp = self._endpoint.set_action_data(id=1)