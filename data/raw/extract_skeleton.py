# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# Extract skeleton description from bvh file

import bpy
import numpy as np
import pickle
import mathutils
import math

ob = bpy.context.object

if ob.type == 'ARMATURE':
    arm = ob.data

print(arm)
        
bone_mapping = {}

num_bones =  len(arm.bones)

# bone's name to id mapping
for id, bone in enumerate(arm.bones):
    bone_mapping[bone.name] = id

# bone's child-to-parent id mapping 
kintree_table = []
for bone in arm.bones:
    
    this_bone_id = bone_mapping[bone.name]
    
    parent = bone.parent
    if parent is None:
        parent_id = -1
    else:
        parent_bone_name = bone.parent.name
        parent_id = bone_mapping[parent_bone_name]
    kintree_table.append([parent_id, this_bone_id])

# bone's LocalRestTransform
J = []
for bone in arm.bones:
    if bone.parent != None:
        bonetrans = bone.parent.matrix_local.inverted()@bone.matrix_local
    else:
        bonetrans = bone.matrix_local
    J.append([bonetrans.row[0].xyzw,bonetrans.row[1].xyzw,bonetrans.row[2].xyzw,bonetrans.row[3].xyzw])

body_data = {}
body_data["J"] = np.array(J)
body_data["kintree_table"] = np.array(kintree_table).T
body_data["name_to_id"] = bone_mapping

with open("data/skeleton.pt", "wb") as f:
    pickle.dump(body_data, f)