from ICCE.interfaces import EnvironmentEndpoint

def main():
    env = EnvironmentEndpoint()
    env.start()
    env.shutdown()

if __name__ == '__main__':
    main()