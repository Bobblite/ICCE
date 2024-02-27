from ICCE.interfaces import ICCEInterface

import numpy as np

class ICCE(ICCEInterface):
    def __init__(self):
        super().__init__(frequency_hz=120)
        self.n_observations = 30
        self.n_actions = 4

    def act(self):
        print('act()')

    def post_sample(self):
        print('post_sample()')

    def post_episode(self):
        print('post_episode()')

def main():
    icce = ICCE()
    icce.run()

if __name__ == '__main__':
    main()