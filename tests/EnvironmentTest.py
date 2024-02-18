import numpy as np
from ICCE.interfaces import EnvironmentInterface

class MyEnvironment(EnvironmentInterface):
    def __init__(self):
        super().__init__()
        self.num_icce = 0
        self.n_observation = 10
        self.n_action = 4
    
    def start(self) -> int:
        print('start() called')

    def sample(self, icce_id: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        data = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)
        return data, 11.0, False, False, {'test':12}
    
    def reset(self):
        print('Reset called')

def main():
    env = MyEnvironment()
    env.register_agent(55)
    env.register_agent(56)
    env.run()
    
if __name__ == '__main__':
    main()