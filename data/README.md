1. open `data/raw/extract_skeleton.blend`, import an bvh, run script `data/raw/extract_skeleont.py` to extract skeleton definition  
2. run `python -m data.extract_bvh_animation` to extract bvh animation to joint's matrix into data/extracted  
3. run `python -m data.prepate_training` to convert bvh animation to axis-angle form for training.  