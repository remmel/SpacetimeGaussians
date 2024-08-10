# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 
import cv2 
import glob 
import tqdm 
import numpy as np 
import shutil
import pickle
import argparse
import cv2
import numpy as np
import os
import json

import natsort
import sys 
import struct
import pickle
from scipy.spatial.transform import Rotation
from pathlib import Path
sys.path.append(".")
from thirdparty.gaussian_splatting.utils.my_utils import rotmat2qvec
from thirdparty.colmap.pre_colmap import * 
from script.helper_pre import getcolmapsingleimundistort
from script.pre_n3d import extractframes
from script.pre_immersive_distorted import softlinkdataset

SCALEDICT = {}


immmersivescaledict = {
    "01_Welder": 1.0,
    "02_Flames": 1.0,
    "04_Truck": 1.0,
    "09_Alexa": 1.0,
    "10_Alexa": 1.0,
    "11_Alexa": 1.0,
    "12_Cave": 1.0
}

Immersiveseven = list(immmersivescaledict.keys())
for scene in Immersiveseven:
    SCALEDICT[scene + "_undist"] = 0.5  # 
    immmersivescaledict[scene + "_undist"] = 0.5


def convertmodel2dbfiles(path, offset=0, scale=1.0):
    projectfolder = path / f"colmap_{offset}"
    manualfolder = projectfolder / "manual"

    manualfolder.mkdir(exist_ok=True)

    savetxt = manualfolder / "images.txt"
    savecamera = manualfolder / "cameras.txt"
    savepoints = manualfolder / "points3D.txt"
    imagetxtlist = []
    cameratxtlist = []

    db_file = projectfolder / "input.db"
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)

    db.create_tables()

    with (path / "models.json").open("r") as f:
        meta = json.load(f)

    for idx, camera in enumerate(meta):
        cameraname = camera['name'] # camera_0001
        view = camera

        focolength = camera['focal_length'] 
        width, height = camera['width'], camera['height']
        principlepoint =[0,0]
        principlepoint[0] = view['principal_point'][0]
        principlepoint[1] = view['principal_point'][1]


        distort1 = view['radial_distortion'][0]
        distort2 = view['radial_distortion'][1]
        distort3 = 0
        distort4 = 0 #view['radial_distortion'][3]


        R = Rotation.from_rotvec(view['orientation']).as_matrix()
        t = np.array(view['position'])[:, np.newaxis]
        w2c = np.concatenate((R, -np.dot(R, t)), axis=1)
        
        colmapR = w2c[:3, :3]
        T = w2c[:3, 3]


        K = np.array([[focolength, 0, principlepoint[0]], [0, focolength, principlepoint[1]], [0, 0, 1]])
        Knew = K.copy()
        
        Knew[0,0] = K[0,0] * float(scale)
        Knew[1,1] = K[1,1] * float(scale)
        Knew[0,2] = view['principal_point'][0] 
        Knew[1,2] = view['principal_point'][1] 


        newfocalx = Knew[0,0]
        newfocaly = Knew[1,1]
        newcx = Knew[0,2]
        newcy = Knew[1,2]


        colmapQ = rotmat2qvec(colmapR)

        imageid = str(idx+1)
        cameraid = imageid
        pngname = f"{cameraname}.png"

        line = f"{imageid} " + " ".join(map(str, colmapQ)) + " " + " ".join(map(str, T)) + f" {cameraid} {pngname}\n"
        imagetxtlist.append(line)
        imagetxtlist.append("\n")

        newwidth = width
        newheight = height
        params = np.array((newfocalx , newfocaly, newcx, newcy,))

        camera_id = db.add_camera(1, newwidth, newheight, params)     # RADIAL_FISHEYE                                                                                 # width and height
        #

        cameraline = f"{idx + 1} PINHOLE {newwidth} {newheight} {newfocalx} {newfocaly} {newcx} {newcy}\n"
        cameratxtlist.append(cameraline)
        image_id = db.add_image(pngname, camera_id,  prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])), prior_t=np.array((T[0], T[1], T[2])), image_id=idx+1)
        db.commit()
    db.close()

    savetxt.write_text("".join(imagetxtlist))
    savecamera.write_text("".join(cameratxtlist))
    savepoints.write_text("")  # Creating an empty points3D.txt file



def imageundistort_no_mapper(video, offsetlist=[0],focalscale=1.0, fixfocal=None):
    with open(video / "models.json", "r") as f:
        meta = json.load(f)

    for idx, camera in enumerate(tqdm.tqdm(meta, desc="Processing Cameras")):
        folder = camera['name'] # camera_0001
        view = camera
        intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                            [0.0, view['focal_length'], view['principal_point'][1]],
                            [0.0, 0.0, 1.0]])
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view['radial_distortion'])[:2]
        map1, map2 = None, None
        for offset in offsetlist:
            imagepath = video / folder / f"{offset}.png"
            imagesavepath = video / f"colmap_{offset}" / "input" / f"{folder}.png"
            if imagesavepath.exists():
                continue

            inputimagefolder = video / f"colmap_{offset}" / "input"
            inputimagefolder.mkdir(exist_ok=True)

            assert imagepath.exists()
            image = cv2.imread(imagepath).astype(np.float32) #/ 255.0
            h, w = image.shape[:2]

            image_size = (w, h)
            knew = np.zeros((3, 3), dtype=np.float32)


            knew[0,0] = focalscale * intrinsics[0,0]
            knew[1,1] = focalscale * intrinsics[1,1]
            knew[0,2] =  view['principal_point'][0] # cx fixed half of the width
            knew[1,2] =  view['principal_point'][1] #
            knew[2,2] =  1.0



            map1, map2 = cv2.fisheye.initUndistortRectifyMap(intrinsics, dis_cef, R=None, P=knew, size=(w, h), m1type=cv2.CV_32FC1)

            undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            undistorted_image = undistorted_image.clip(0,255.0).astype(np.uint8)

            cv2.imwrite(imagesavepath, undistorted_image)
        else:
            print("already exists")







if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--videopath", default="", type=str)
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=50, type=int)
    args = parser.parse_args()

    videopath = Path(args.videopath)
    startframe = args.startframe
    endframe = args.endframe
    if startframe >= endframe:
        print("start frame must smaller than end frame")
        quit()
    if startframe < 0 or endframe > 300:
        print("frame must in range 0-300")
        quit()
    if not videopath.exists():
        print("path not exist")
        quit()

    srcscene = videopath.name
    if srcscene not in Immersiveseven:
        print("scene not in Immersiveseven", Immersiveseven)
        print("Please check if the scene name is correct")
        quit()
    

    if "04_Trucks" == srcscene:
        print('04_Trucks')
        if endframe > 150:
            endframe = 150 

    postfix  = "_undist" # undistored cameras

    scene = srcscene + postfix
    dstpath = videopath.with_name(videopath.name + postfix)

    scale = immmersivescaledict[scene]

    videoslist = list(videopath.glob("*.mp4"))
    for v in tqdm.tqdm(videoslist):
        extractframes(v, startframe, endframe)

    softlinkdataset(str(videopath), str(dstpath))

    imageundistort_no_mapper(dstpath, offsetlist=list(range(startframe,endframe)),focalscale=scale, fixfocal=None)


    try:
        for offset in tqdm.tqdm(range(startframe, endframe), desc="convertmodel2dbfiles"):
            convertmodel2dbfiles(dstpath, offset=offset, scale=scale)
    except:
        print("create colmap input failed, better clean the data and try again")
        quit()

    for offset in range(startframe, endframe):
        getcolmapsingleimundistort(dstpath, offset=offset)
