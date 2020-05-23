# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# batch script to extract all animation in bvh files

import os
import tqdm
mocap_path = "data/raw/"
mocap_files = []
for r, d, f in os.walk(mocap_path):
    mocap_files.extend(["{}/{}".format(r,file) for file in f if file.endswith(".bvh")])
print(mocap_files)

command_template = "blender --background -P data/bpy_import_bvh_and_convert.py  --bvh {}"

loop = tqdm.tqdm(range(0, len(mocap_files)))
for i in loop:
    mocap_file = mocap_files[i]
    print("process ", mocap_file)
    os.system(command_template.format(mocap_file))
