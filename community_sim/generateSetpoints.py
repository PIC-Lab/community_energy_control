import os
import json
from pathlib import Path
import random
import argparse

def Main():
    '''Creates a json file containing the heating and cooling setpoints and deadband for each building'''
    parser = argparse.ArgumentParser('Generate Setpoints')
    parser.add_argument('-f', '--file', help='File path to the building mapping json file')
    args = parser.parse_args()

    # Valid values
    heating = list(range(65,73))
    cooling = list(range(70, 78))
    deadband = list(range(1,6))
    # Check if json already exists
    fileName = 'buildingSetpoints.json'
    if os.path.isfile(fileName):
        print('Json file already exists. Running this script will clear all saved values.')
        userInput = input('Do you still want to continue? y/n: ')
        if (userInput == 'n') or (userInput == 'N'):
            print('Stopping script')
            return
        
    tempDict = {}
        
    if args.file is None: 
        buildingDir = Path('building_models')

        # Create dictionary
        tempDict = {}

        for file in buildingDir.iterdir():
            if file.is_dir():
                db = random.choice(deadband) 
                hs = random.choice(heating)
                cs = random.choice([x for x in cooling if x >= hs + db])
                tempDict[file.name] = {'heatSP': (hs-32)*5/9,
                                    'coolSP': (cs-32)*5/9,
                                    'deadband': db*5/9}
    else:
        with open(args.file) as jsonData:
            aliasMap = json.load(jsonData)
        
        for key, value in aliasMap.items():
            for i, building in enumerate(value['buildings']):
                for j in range(0,value['number'][i]):
                    db = random.choice(deadband) 
                    hs = random.choice(heating)
                    cs = random.choice([x for x in cooling if x >= hs + db])
                        
                    if j == 0:
                        name = building
                    else:
                        name = building + '_' + str(j)
                    
                    tempDict[name] = {'heatSP': (hs-32)*5/9,
                                    'coolSP': (cs-32)*5/9,
                                    'deadband': db*5/9}
            
    tempDict = dict(sorted(tempDict.items(), key=mySort))

    with open(fileName, 'w') as fi:
        json.dump(tempDict, fi)
    print('Json file created')

def mySort(n):
    if n[0].isnumeric():
        temp = int(n[0])
    else:
        index = n[0].find('_')
        temp = int(n[0][0:index]) + 0.1 * int(n[0][index+1:])
    return temp

if __name__ == '__main__':
    Main()