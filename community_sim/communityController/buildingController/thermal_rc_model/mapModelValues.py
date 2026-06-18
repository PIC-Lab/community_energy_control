import json

def Main():
    with open('buildings_tuned.json') as fp:
        models = json.load(fp)

    with open('../../../configs/indexMapping.json') as fp:
        sensorIdxMap = json.load(fp)

    models_mapped = {}
    for key, value in sensorIdxMap.items():
        models_mapped[key] = models[value]

    with open('buildings_tuned_map.json', 'w') as fp:
        json.dump(models_mapped, fp)

if __name__ == '__main__':
    Main()