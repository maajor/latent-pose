
import bpy
import numpy as np
import pickle

ob = bpy.context.object

if ob.type == 'ARMATURE':
    arm = ob.data

print(arm)
        
bone_mapping = {}

num_bones =  len(arm.bones)

for id, bone in enumerate(arm.bones):
    bone_mapping[bone.name] = id
    
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

#print(kintree_table)
J = []
for bone in arm.bones:
    bone_pos = bone.head_local
    J.append([bone_pos.x, bone_pos.y, bone_pos.z])
    #print(bone.name, bone.matrix) # bone in object space


body_data = {}
body_data["J"] = np.array(J)
body_data["kintree_table"] = np.array(kintree_table).T
body_data["name_to_id"] = bone_mapping

with open("skeleton.pt", "wb") as f:
    pickle.dump(body_data, f)