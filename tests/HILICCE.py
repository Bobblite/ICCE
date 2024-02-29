from ICCE.interfaces import ICCEInterface
from InputManager import InputManager

import numpy as np
from pynput.keyboard import Key

class HILICCE(ICCEInterface):
    def __init__(self):
        super().__init__(frequency_hz=120, agent_hint=0)
        self.n_observations = 30
        self.n_actions = 4
        self.inputManager = InputManager()
        

    def post_sample(self, observation: np.ndarray, reward: float):
        pass

    def post_episode(self):
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.handle_inputs()

    def handle_inputs(self):
        # Reset actions
        action = np.zeros(shape=(4), dtype=np.float32)

        # Handle inputs
        if (self.inputManager.isPressed(Key.left)):
            action[0] -= 1.0
        if (self.inputManager.isPressed(Key.right)):
            action[0] += 1.0
        if (self.inputManager.isPressed(Key.down)):
            action[1] -= 1.0
        if (self.inputManager.isPressed(Key.up)):
            action[1] += 1.0
        if (self.inputManager.isPressed(Key.space)):
            action[3] = 1.0
        
        return action

def main():
    icce = HILICCE()
    icce.inputManager.start()
    icce.run()
    icce.inputManager.stop()

if __name__ == '__main__':
    main()