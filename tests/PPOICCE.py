from ICCE.interfaces import ICCEInterface

import numpy as np

class PPOICCE(ICCEInterface):
    def __init__(self):
        super().__init__(frequency_hz=120)
        self.n_observations = 30
        self.n_actions = 4

    def post_sample(self, observation: np.ndarray, reward: float):
        print('post_sample()')

    def post_episode(self):
        print('post_episode()')

    def act(self, observation: np.ndarray) -> np.ndarray:
        print('act()')
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        

def main():
    icce = PPOICCE()
    icce.run()

if __name__ == '__main__':
    main()