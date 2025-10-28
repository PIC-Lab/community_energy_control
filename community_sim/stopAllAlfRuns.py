from alfalfa_client.alfalfa_client import AlfalfaClient
from time import sleep, time
import argparse
from typing import List, Union
from logging import Logger
from datetime import datetime
import requests

from alfalfa_client.lib import (
    AlfalfaAPIException,
    AlfalfaException,
    AlfalfaClientException,
    parallelize
)

def Main():
    parser = argparse.ArgumentParser("Alfalfa Client Extended")
    parser.add_argument('-n', '--name', default='stop', help="Name of desired operation")
    args = parser.parse_args()

    ac = AlfalfaClientCustom(host='http://localhost')

    if args.name == 'stop':
        run_ids = ac.get_active_runs()
        ac.stop(run_ids)
        print(f"Stopped {len(run_ids)} runs")
    elif args.name == 'delete_complete':
        run_ids = ac.get_complete_runs()
        for run_id in run_ids:
            ac.delete_run(run_id)
        print(f"Deleted {len(run_ids)} runs")
    elif args.name == 'delete_idle':
        run_ids = ac.get_idle_runs()
        ac.stop(run_ids)
        ac.delete_run(run_ids)
        print(f"Deleted {len(run_ids)} runs")
    elif args.name == 'delete_all':
        run_ids = ac.get_active_runs()
        ac.stop(run_ids)
        run_ids = ac.get_all_runs()
        ac.delete_run(run_ids)
        print(f"Deleted {len(run_ids)} runs")

RunID = str

class AlfalfaClientCustom(AlfalfaClient):
    '''
    '''
    def __init__(self, host: str = 'http://localhost', api_version: str = 'v2', logger: Logger = None):
        '''
        Constructor
        Parameters:
            host: (str) url for host of alfalfa web server
        '''
        super().__init__(host, api_version)

        self.logger = logger

    def get_all_runs(self):
        '''
        '''
        response = self._request("runs/", method="GET")
        response_body = response.json()["payload"]

        run_ids = []
        for run in response_body:
            run_ids.append(run["id"])

        return run_ids

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

    def check_idle(self, run_ids:list[str]):
        '''
        '''
        currentTime = self.get_sim_time(run_ids)
        sleep(120)
        newTime = self.get_sim_time(run_ids)

        return currentTime == newTime

    @parallelize
    def delete_run(self, run_id):
        '''
        Delete a specified run
        :param run_id: (str) id of run to delete
        '''
        response = self._request(f"runs/{run_id}", method="DELETE")

        assert response.status_code == 204, "Got wrong status_code from alfalfa"

    @parallelize
    def wait(self, run_id: Union[RunID, List[RunID]], desired_status: str, timeout: float =600) -> None:
        '''
        Wait for a run to have a certain status or timeout with error

        :param run_id: id of run of list of ids
        :param desired_status: status to wait for
        :param timout: timeout length in seconds
        '''

        start_time = time()
        previous_status = None
        current_status = None
        while time() - timeout < start_time:
            try:
                current_status = self.status(run_id)
            except AlfalfaAPIException as e:
                if e.response.status_code != 404:
                    raise e
                
            if current_status == "ERROR":
                error_log = self.get_error_log(run_id)
                raise AlfalfaException(error_log)
            
            if current_status != previous_status:
                if not(self.logger is None):
                    self.logger.info("Desired status: {}\t\tCurrent status: {}".format(desired_status, current_status))
                else:
                    print("Desired status: {}\t\tCurrent status: {}".format(desired_status, current_status))
                previous_status = current_status
            if current_status == desired_status.upper():
                return
            sleep(2)
        raise AlfalfaClientException(f"'wait' timed out waiting for status: '{desired_status}', curren status: '{current_status}'")
    
    @parallelize
    def advance(self, run_id: Union[RunID, List[RunID]]) -> None:
        '''
        Advance a run 1 timestep

        :param run_id: id of run or list of ids
        '''
        try:
            self._request(f"runs/{run_id}/advance")
            return True
        except requests.exceptions.ConnectionError:
            return False



if __name__ == '__main__':
    Main()