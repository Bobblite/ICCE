import numpy as np
from ICCE.interfaces import EnvironmentInterface

class MyEnvironment(EnvironmentInterface):
    def __init__(self):
        super().__init__(expected_agents=2)
        self.num_icce = 0
        self.n_observations = 10
        self.n_actions = 4

    def generate_id(self) -> (int, int):
        id = self.num_icce
        agent_id = id
        self.num_icce+=1
        return id, agent_id
    
    def sample(self, icce_id: int) -> (np.ndarray, float, bool, bool, dict):
        data = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
        return data, 11, False, False, {'test':12}
    
    def reset(self) -> None:
        print('Reset called')
        pass

def main():
    env = MyEnvironment()
    env.run()
    
if __name__ == '__main__':
    main()