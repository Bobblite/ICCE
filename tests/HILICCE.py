from ICCE.interfaces import ICCEInterface
from InputManager import InputManager

import numpy as np
from pynput.keyboard import Key

class HILICCE(ICCEInterface):
    def __init__(self):
        super().__init__(frequency_hz=120)
        self.n_observations = 30
        self.n_actions = 4
        self.action = np.zeros(shape=(4), dtype=np.float32)
        self.inputManager = InputManager()
        

    def post_sample(self):
        pass

    def post_episode(self):
        pass

    def act(self):
        self.action = np.zeros(shape=(4), dtype=np.float32)
        self.handle_inputs()

    def handle_inputs(self):
        # Reset actions
        action = np.zeros(4)

        # Handle inputs
        if (self.inputManager.isPressed(Key.left)):
            self.action[0] -= 1.0
        if (self.inputManager.isPressed(Key.right)):
            self.action[0] += 1.0
        if (self.inputManager.isPressed(Key.down)):
            self.action[1] -= 1.0
        if (self.inputManager.isPressed(Key.up)):
            self.action[1] += 1.0
        if (self.inputManager.isPressed(Key.space)):
            self.action[3] = 1.0

def main():
    icce = HILICCE()
    icce.inputManager.start()
    icce.run()
    icce.inputManager.stop()

if __name__ == '__main__':
    main()