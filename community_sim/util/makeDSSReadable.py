from pathlib import Path
import fnmatch
import re
import shutil

def Main():
    '''
    Script for converting the names set in the dss files by URBANopt into readable, non-uuid
    names. Creates a mapping dict by reading the transformer and load files, then converts the
    names in all files.
    
    If I run into another issue with the dss files that is impossible to troubleshoot
    because OpenDSSDirect has no error messages and it turns out it was just another case of
    mismatched ids, a problem that would be trivial to notice if the buses used sane,
    informative names instead of uuids, I'm going to lose it.
    '''

    networkDir = Path('network_model')

    uuidMap = {}

    unknownBusName = 500
    # Create key values pairs from load file
    with open(networkDir / 'Loads.dss') as file:
        for line in file:
            uuids = GetUUIDs(line)
            for id in uuids:
                index = line.find('bus1='+id)
                if index == -1:
                    print(f'UUID {id} was present in load file but not used as expected.')
                    continue
                startIdx = line.find('Load.') + 5
                endIdx = line[startIdx:].find(' ') + startIdx
                uuidMap[id] = f'Bus{line[startIdx:endIdx]}'

    # Create key values pairs from transformer file
    with open(networkDir / 'Transformers.dss') as file:
        i = 1
        for line in file:
            uuids = GetUUIDs(line)
            if len(uuids) == 0:
                continue

            j = 1
            for id in uuids:
                index = line.find('Transformer.'+id)
                if index == -1:
                    if not(id in uuidMap.keys()):
                        uuidMap[id] = f'Bus_Xfmr{i}_{j}'
                        j += 1
                    continue
                uuidMap[id] = f'Xfmr{i}'
            i += 1

    # Create key values pairs from line file
    with open(networkDir / 'Lines.dss') as file:
        for line in file:
            uuids = GetUUIDs(line)
            for id in uuids:
                if line.find('Line.'+id) != -1:
                    nameId = id
                    continue
                if line.find('bus1='+id) != -1:
                    if not(id in uuidMap.keys()):
                        uuidMap[id] = f'Bus_Distr{unknownBusName}'
                        unknownBusName += 1
                    bus1Id = uuidMap[id]
                    continue
                if line.find('bus2='+id) != -1:
                    if not(id in uuidMap.keys()):
                        uuidMap[id] = f'Bus_Distr{unknownBusName}'
                        unknownBusName += 1
                    bus2Id = uuidMap[id]
                    continue
            
            # Special case where bus name isn't a uuid (source)
            if bus1Id is None:
                startIdx = line.find('bus1=') + 5
                endIdx = line[startIdx:].find('.') + startIdx
                bus1Id = line[startIdx:endIdx]
            if bus2Id is None:
                startIdx = line.find('bus2=') + 5
                endIdx = line[startIdx:].find('.') + startIdx
                bus1Id = line[startIdx:endIdx]

            uuidMap[nameId] = f'Line-{bus1Id}-{bus2Id}'

    for fileName in networkDir.iterdir():
        if fileName.is_dir():
            continue
        print(fileName)
        with open(fileName) as old, open(networkDir / 'temp.dss', 'w') as new:
            for line in old:
                uuids = GetUUIDs(line)
                for id in uuids:
                    line = line.replace(id, uuidMap[id])
                new.write(line)
        shutil.move(networkDir / 'temp.dss', fileName)
    

def GetUUIDs(inputStr):
    '''
    Takes one line of a dss file as input and returns a list of the uuids
    '''
    strList = re.split(r'\s|=|\.', inputStr)
    return fnmatch.filter(strList, '????????-????-????-????-????????????')

if __name__ == '__main__':
    Main()