# from werkzeug.utils import secure_filename
import os.path
import subprocess
from typing import List
import shutil
import uvicorn
import pathlib
from utils_ import get_logger
from fastapi import File, UploadFile, FastAPI
from fastapi.responses import FileResponse

logger = get_logger('app', 'log.txt')

DIR_PHOTOS = 'photos'
DIR_WEIGHTS = 'weights'
FILENAME_WEIGHTS = 'model.pt'
FILENAME_DETECTED_ZIP = 'detected'

import os
cwd = os.getcwd()
print(cwd)



# DIR_TILE = 'tiles'
# START_TILE_SIZE = 1280
# FINISH_TILE_SIZE = 1280
# JPEG_QUALITY = 95

app_api = FastAPI()


def zip_directory(dir_to_zip, output_filename):
    shutil.make_archive(output_filename, 'zip', dir_to_zip)


def detect():
    filepath_weights = os.path.join(DIR_WEIGHTS, FILENAME_WEIGHTS)
    res = subprocess.run(['python3', 'detect.py', '--weights', filepath_weights, '--source', DIR_PHOTOS, '--imgsz', '1280', '--save-txt', '--save-conf', '--hide-conf'], capture_output=True, text=True)
    logger.info(f'yolo finished with {res}')


@app_api.post("/detect/")
async def create_upload_files(files: List[UploadFile]):

    """
    files =[('files', open(f'tiled/{i}', 'rb')) for i in os.listdir('tiled')]
    files += [('files', open('/home/anon/coding/yolov5/runs/train/exp17/weights/best.pt', 'rb'))]


    r = requests.post('http://127.0.0.1:8003/detect',
                      files=files)

    with open("detected.zip", "wb") as f:
        f.write(r.content)

    shutil.unpack_archive("detected.zip", 'detected')
    """

    pathlib.Path(DIR_PHOTOS).mkdir(parents=True, exist_ok=True)
    pathlib.Path(DIR_WEIGHTS).mkdir(parents=True, exist_ok=True)

    for file in files:
        contents = await file.read()
        filename = file.filename
        if filename.endswith('.pt'):
            filepath = os.path.join(DIR_WEIGHTS, FILENAME_WEIGHTS)
            with open(filepath, 'wb') as f:
                f.write(contents)
        else:
            filepath = os.path.join(DIR_PHOTOS, filename)
            with open(filepath, 'wb') as f:
                f.write(contents)

    detect()
    shutil.make_archive(FILENAME_DETECTED_ZIP, 'zip', 'runs/detect/exp')
    shutil.rmtree('runs/detect/exp')
    shutil.rmtree(DIR_PHOTOS)

    return FileResponse(FILENAME_DETECTED_ZIP + '.zip')


if __name__ == '__main__':
    uvicorn.run(app_api, host='0.0.0.0', port=8003)

