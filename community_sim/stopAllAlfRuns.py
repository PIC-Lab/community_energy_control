from alfalfa_client.alfalfa_client import AlfalfaClient
import requests

def Main():
    ac = AlfalfaClientCustom(host='http://localhost')

    run_ids = ac.get_active_runs()
    ac.stop(run_ids)
    print(f"Stopped {len(run_ids)} runs")

class AlfalfaClientCustom(AlfalfaClient):
    '''
    '''
    def __init__(self, host: str = 'http://localhost'):
        '''
        Constructor
        Parameters:
            host: (str) url for host of alfalfa web server
        '''
        super().__init__(host)

    def get_active_runs(self):
        '''
        Returns a list of run ids that are currently active
        Returns: list of run ids
        '''
        response = self._request("runs/", method="GET")
        response_body = response.json()["payload"]

        run_ids = []
        for run in response_body:
            if run["status"] == "RUNNING":
                run_ids.append(run["id"])

        return run_ids

if __name__ == '__main__':
    Main()