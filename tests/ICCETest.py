from ICCE.interfaces import ICCEEndpoint

def main():
    icce = ICCEEndpoint()
    handshake_resp = icce.handshake_and_validate(id=1)
    start_resp = icce.start_simulation(id=1)
    env_data = icce.get_env_data(id=1)
    print(f'Received status: {env_data.status} | episode: {env_data.episode}')
    action_resp = icce.set_action_data(id=1)

if __name__ == '__main__':
    main()