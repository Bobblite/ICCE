from ICCE.interfaces import EnvironmentEndpoint
import time

def main():
    env = EnvironmentEndpoint()
    env.start()

    time.sleep(5)
    env.shutdown()
    
if __name__ == '__main__':
    main()