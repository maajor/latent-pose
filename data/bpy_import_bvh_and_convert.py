# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# blender python script to import an bvh and extract pose

import bpy
import numpy as np
import pickle
import sys

argv = sys.argv
print(argv)
filename = argv[argv.index("--bvh") + 1]
print(filename)
bpy.ops.import_anim.bvh(filepath=filename)


ob = bpy.context.object
sce = bpy.context.scene

with open("data/skeleton.pt", "rb") as f:
    mapping = pickle.load(f)["name_to_id"]

# https://blender.stackexchange.com/questions/27889/how-to-find-number-of-animated-frames-in-a-scene-via-python
def get_keyframes(obj):
    keyframes = []
    anim = obj.animation_data
    if anim is not None and anim.action is not None:
        for fcu in anim.action.fcurves:
            for keyframe in fcu.keyframe_points:
                x, y = keyframe.co
                if x not in keyframes:
                    keyframes.append(int(x))
    return keyframes

if ob.type == 'ARMATURE':
    armature = ob
    
if armature != None:
    keys = get_keyframes(armature)
    anim_data = np.zeros((len(keys)-1, len(mapping.keys()),3,3))

    for f in range(1, keys[-1]):
        sce.frame_set(f)
        
        for pbone in armature.pose.bones:
            mat_local = np.array(pbone.matrix_basis)[:3,:3]
            pbone_id = mapping[pbone.name]
            anim_data[f-1,pbone_id,:,:] = mat_local

    print(anim_data.shape)

    with open("data/extracted/{}.pkl".format(armature.name), "wb") as f:
        pickle.dump(anim_data, f)
else:
    print("Cannot find armature")