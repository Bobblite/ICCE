import numpy as np
from ICCE.interfaces import EnvironmentInterface
from pyflyt_dogfight import SimulationListener

class Environment(EnvironmentInterface):
    def __init__(self):
        super().__init__(max_episodes=10, debug=True)

        # Set input/output sizes
        self.n_observation = 30
        self.n_action = 4

        # Simulation listener API
        self._sim_listener = SimulationListener()
        self.i = 0
    
    def start(self) -> int:
        print('start() called')

    def sample(self, agent_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        response = self._sim_listener.sample(agent_id)
        obs = np.frombuffer(buffer=response.observation, dtype=np.float64)
        reward = response.reward
        term = response.terminated
        trunc = response.truncated
        return obs, reward, term, trunc, {'test':True}
    
    def reset(self):
        response = self._sim_listener.reset(iterations=1)
        print('reset env: ', self.i)
        self.i+=1
        return response.status
    
    def act(self, agent_id, action):
        self._sim_listener.act(agent_id, action.tobytes())


def main():
    env = Environment()
    env.register(0)     # HIL Agent
    env.register(1)     # ICCE Agent
    env.run()
    
if __name__ == '__main__':
    main()