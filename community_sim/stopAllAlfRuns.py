from alfalfa_client.alfalfa_client import AlfalfaClient
from time import sleep, time
import argparse

def Main():
    parser = argparse.ArgumentParser("Alfalfa Client Extended")
    parser.add_argument('-n', '--name', default='stop', help="Name of desired operation")
    args = parser.parse_args()

    ac = AlfalfaClient(host='http://localhost')

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

if __name__ == '__main__':
    Main()