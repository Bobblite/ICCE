from ICCE.interfaces import EnvironmentInterface

class MyEnvironment(EnvironmentInterface):
    def __init__(self):
        super().__init__()
        self.num_icce = 0
        self.n_observations = 10
        self.n_actions = 4

    def generate_id(self):
        self.add_icce(self.num_icce, self.num_icce)
        self.num_icce += 1

def main():
    env = MyEnvironment()
    env.run()
    
if __name__ == '__main__':
    main()