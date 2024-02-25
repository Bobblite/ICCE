import numpy as np

from pyflyt_dogfight import DogfightEnv, SimulationPublisher
import threading
import time

class Simulation:
    def __init__(self):
        # Simulation
        self.env = DogfightEnv(render=True)

        # Simulation data
        self.observation = np.zeros(shape=(2, 30), dtype=np.float64)
        self.rewards = np.zeros(shape=(2), dtype=np.float64)
        self.truncated = np.full(shape=(2), fill_value=False, dtype=bool)
        self.terminated = np.full(shape=(2), fill_value=False, dtype=bool)
        self.info = None # TODO

        # API
        self._sim_publisher = SimulationPublisher(self.sample, self.on_reset)
        self.reset_flag = False

        # Mutex Lock
        self._lock = threading.Lock()

        # Initial reset
        self.reset()

    def start(self):
        # Start API
        self._sim_publisher.start()
        # Iterate though simulation
        while True:
            # Reset if flag set
            if self.reset_flag:
                self.reset()

            # Set simulation data
            actions = np.zeros(shape=(2, 4))
            actions[:, -1] = 1.0
            obs, rewards, terminated, truncated, info = self.env.step(actions)

            # self.observation, self.rewards, self.terminated, self.truncated, self.info = self.env.step(actions)
            
            with self._lock:
                self.observation = obs.copy()
                self.rewards = rewards.copy()
                self.terminated = terminated.copy()
                self.truncated = truncated.copy()
                self.info = info.copy()

    def reset(self):
        # Reset env
        obs, info = self.env.reset()

        with self._lock:
            print('Resetting')
            # Set initial simulation data
            self.observation = obs
            self.info = info
            self.truncated = np.full(shape=(2), fill_value=False, dtype=bool)
            self.terminated = np.full(shape=(2), fill_value=False, dtype=bool)

            self.reset_flag = False
        return True
    
    def sample(self, agent_id):
        # Get data
        with self._lock:
            obs = self.observation[agent_id].copy()
            reward = self.rewards[agent_id]
            term = self.terminated[agent_id]
            trunc = self.truncated[agent_id]

        return obs.tobytes(), reward, term, trunc

    def on_reset(self):
        print('Set reset flag')
        with self._lock:
            self.reset_flag = True
            
def main():
    sim = Simulation()
    sim.start()

if __name__ == '__main__':
    main()