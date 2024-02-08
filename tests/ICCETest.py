from ICCE.interfaces import ICCEEndpoint

def main():
    icce = ICCEEndpoint()
    env_data = icce.get_env_data(id=1)
    print(f'Received status: {env_data.status} | episode: {env_data.episode}')

if __name__ == '__main__':
    main()