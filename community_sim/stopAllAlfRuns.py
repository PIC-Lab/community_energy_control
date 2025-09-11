from alfalfa_client.alfalfa_client import AlfalfaClient
from alfalfa_client.lib import parallelize
from time import sleep, time
import argparse

def Main():
    parser = argparse.ArgumentParser("Alfalfa Client Extended")
    parser.add_argument('-n', '--name', default='stop', help="Name of desired operation")
    args = parser.parse_args()

    ac = AlfalfaClientCustom(host='http://localhost')

    if args.name == 'stop':
        run_ids = ac.get_active_runs()
        ac.stop(run_ids)
        print(f"Stopped {len(run_ids)} runs")
    elif args.name == 'delete':
        run_ids = ac.get_complete_runs()
        for run_id in run_ids:
            ac.stop(run_id)
        print(f"Deleted {len(run_ids)} runs")


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
    
    def get_complete_runs(self):
        '''
        Returns a list of run ids that are currently complete
        Returns: list of run ids
        '''
        response = self._request("runs/", method="GET")
        response_body = response.json()["payload"]

        run_ids = []
        for run in response_body:
            if run["status"] == "COMPLETE":
                run_ids.append(run["id"])

        return run_ids
    
    def get_idle_runs(self):
        '''
        Returns a list of run ids that are running, but are not being actively advanced
        Return: list of run ids
        '''
        response = self._request("runs/", method="GET")
        response_body = response.json()["payload"]

        run_ids = []
        for run in response_body:
            run_ids.append(run["id"])
        idle_ids = self.check_idle(run_ids)
        return [run_ids[i] for i in range(0, len(run_ids)) if idle_ids[i]]

    @parallelize
    def check_idle(self, run_id):
        '''
        '''
        currentTime = self.get_sim_time(run_id)
        sleep(120)
        newTime = self.get_sim_time(run_id)

        if currentTime == newTime:
            return True
        else:
            return False

    def delete_run(self, run_id):
        '''
        Delete a specified run
        :param run_id: (str) id of run to delete
        '''
        response = self._request(f"runs/{run_id}", method="DELETE")

        assert response.status_code == 204, "Got wrong status_code from alfalfa"

if __name__ == '__main__':
    Main()