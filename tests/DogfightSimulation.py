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
        self.action = np.zeros(shape=(2, 4), dtype=np.float32)

        # Workaround of reward setting to 0 on terminate
        self.done = False

        # API
        self._sim_publisher = SimulationPublisher(sample_cb=self.on_sample, act_cb=self.on_act, reset_cb=self.on_reset)
        self.reset_flag = True

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

            if self.done:
                continue

            # Set simulation data
            # actions = np.zeros(shape=(2, 4))
            # actions[:, -1] = 1.0

            # Step environment
            obs, rewards, terminated, truncated, info = self.env.step(self.action)

            if any(terminated) or any(truncated):
                self.done = True

            with self._lock:
                self.observation = obs.copy()
                self.rewards = rewards.copy()
                self.terminated = terminated.copy()
                self.truncated = truncated.copy()
                self.info = info.copy()

    def reset(self):
        # Reset env
        with self._lock:
            obs, info = self.env.reset()

            # Set initial simulation data
            self.observation = obs
            self.info = info
            self.truncated = np.full(shape=(2), fill_value=False, dtype=bool)
            self.terminated = np.full(shape=(2), fill_value=False, dtype=bool)
            self.reset_flag = False
            self.done = False
            return True
    
    def on_sample(self, agent_id):
        # Get data
        with self._lock:
            obs_bytes = self.observation[agent_id].copy().tobytes()
            reward = self.rewards[agent_id]
            term = self.terminated[agent_id]
            trunc = self.truncated[agent_id]
        return obs_bytes, reward, term, trunc

    def on_act(self, agent_id, action_bytes):
        self.action[agent_id] = np.frombuffer(buffer=action_bytes, dtype=np.float32)

    def on_reset(self):
        with self._lock:
            self.reset_flag = True
            
def main():
    sim = Simulation()
    sim.start()

if __name__ == '__main__':
    main()