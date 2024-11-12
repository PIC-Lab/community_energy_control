import os
import json
from pathlib import Path
import argparse
import shutil

def Main():
    '''Maps a saved run from simulation indices to sensor indices, which is needed for co-sim and deployment'''
    parser = argparse.ArgumentParser('Prepare Run for Deployment')
    parser.add_argument('runName', help='Name of the saved run that is selected for deployment')
    args = parser.parse_args()

    deploymentDir = Path('deployModels')
    savedRun = Path('savedRuns/' + args.runName)
    deployedRun = deploymentDir / args.runName

    # Create folder if it doesn't exist
    if not(os.path.exists(deploymentDir)):
        os.mkdir(deploymentDir)

    # Copy saved run into deployment folder
    shutil.copytree(savedRun, deployedRun, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.png', '*.csv'))

    with open('../../../indexMapping.json') as fp:
        sensorIdxMapping = json.load(fp)        # Map sensor indices to simulation indices
    simIdxMapping = {v: k for k, v in sensorIdxMapping.items()}     # Map simulation indices to sensor indices

    # Update indices for buildingThermal
    src = savedRun / 'buildingThermal'
    dst = deployedRun / 'buildingThermal'
    shutil.rmtree(dst)
    os.mkdir(dst)
    for file in (src).iterdir():
        if file.is_dir():
            shutil.copytree(file, dst / simIdxMapping[file.name], ignore=shutil.ignore_patterns('*.png', '*.csv'))

    # Update indices for controller
    src = savedRun / 'controller'
    dst = deployedRun / 'controller'
    shutil.rmtree(dst)
    os.mkdir(dst)
    for file in (src).iterdir():
        if file.is_dir():
            shutil.copytree(file, dst / simIdxMapping[file.name], ignore=shutil.ignore_patterns('*.png', '*.csv', '*.gif'))

    # Update indices for norm
    src = savedRun / 'norm'
    dst = deployedRun / 'norm'
    shutil.rmtree(dst)
    os.mkdir(dst)
    for file in (src).iterdir():
        if file.is_dir():
            shutil.copytree(file, dst / simIdxMapping[file.name])

if __name__ == '__main__':
    Main()