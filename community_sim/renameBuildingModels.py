from pathlib import Path
import shutil

def Main():
    modelDir = Path('building_models')
    
    for dirName in modelDir.iterdir():
        if not(dirName.is_dir()):
            continue
        print(dirName)

        with open(dirName / f'models/{dirName.name}.osm') as file:
            fileData = file.read()

        fileData = fileData.replace('Building 1', f'Building {dirName.name}')

        with open(dirName / f'models/{dirName.name}.osm', 'w') as file:
            file.write(fileData)


if __name__ == '__main__':
    Main()