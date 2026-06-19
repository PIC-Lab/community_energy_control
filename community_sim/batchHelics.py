import subprocess
import json
import traceback
from time import sleep
import shutil

def Main():
    batches = [
        {'start': "01/02/24 00:00",
         'batDischargeMode': "loadFollow"},
        {'start': "01/02/24 00:00",
         'batDischargeMode': "bulk"},
        {'start': "07/01/24 00:00",
         'batDischargeMode': "loadFollow"},
        {'start': "07/01/24 00:00",
         'batDischargeMode': "bulk"},
    ]
    skip = 0

    with open('configs/simParams.json') as fp:
        origParams = json.load(fp)

    param_list = []
    params = origParams.copy()

    params['start'] = "07/01/24 00:00"
    params['step'] = '1m'
    params['duration'] = '7D'
    params['warmup'] = '5m'
    params['testCase'] = 'MPC_alt'
    params['controlledAliases'] = ["2", "3", "4", "5", "9", "10", "12", "15", "17", "22", "24", "25", "26", "27", "28"]
    params['logLevel'] = 'INFO'
    params['resultLevel'] = 'NORMAL'
    params['resultsDir'] = f"results/batch/"
    params['controllerRun'] = 'return_7'
    params['nstepsCoord'] = 48
    params['stepSizeCoord'] = 5
    params['nstepsBuild'] = 60
    params['stepSizeBuild'] = 1
    params["controlledLoads"] = ["hvac", "battery"]
    params["batCoveredLoads"] = ["base", "waterHeater", "hvac"]
    params["batDischargeMode"] = "loadFollow"
    params["note"] = ""

    batchNum = 0

    for b in batches:
        for key, value in b.items():
            params[key] = value
            params['resultsDir'] = f"results/batch_{batchNum}/"
                
        param_list.append(params.copy())
        batchNum += 1

    print(f"Running batch co-sim with {batchNum} configs")

    for i, param in enumerate(param_list):
        if i < skip:
            print(f"Skipping config {i}")
            continue
        print(f"Starting config {i}")
        with open('configs/simParams.json', "w") as fp:
            json.dump(param, fp)

        result = subprocess.run(['sh', './helicsRunner.sh'])
        if result.returncode != 0:
            raise RuntimeError

        sleep(10)

        result = subprocess.run(['python', 'stopAllAlfRuns.py'])
        result = subprocess.run(['python', 'stopAllAlfRuns.py', '-n', 'delete_all'])

        sleep(10)

    with open('configs/simParams.json', "w") as fp:
        json.dump(origParams, fp)

if __name__ == '__main__':
    Main()