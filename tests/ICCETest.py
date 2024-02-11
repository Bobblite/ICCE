from ICCE.interfaces import ICCEInterface

class MyICCE(ICCEInterface):
    def __init__(self):
        super().__init__()

        self.n_observations = 10
        self.n_actions = 4

def main():
    icce = MyICCE()
    icce.run()
    

if __name__ == '__main__':
    main()