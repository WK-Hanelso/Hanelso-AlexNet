import os
import shutil

target_dir = "DATA/PokemonData/images"
dest_dir = "DATA/PokemonData/new"


tmp_list = os.listdir( target_dir )

if os.path.exists( dest_dir ) != True:
    os.mkdir( dest_dir )

for cls_name in tmp_list:
    cls_dir = os.path.join( target_dir, cls_name)
    cls_list = os.listdir( cls_dir)

    destcls_dir = os.path.join( dest_dir, cls_name )
    if os.path.exists( destcls_dir ) != True:
        os.mkdir( destcls_dir )

    for idx, img_name in enumerate(cls_list):
        img_dir = os.path.join( cls_dir, img_name)
        ext = img_name.split('.')[-1]
        destimg_dir = os.path.join( destcls_dir, cls_name + "_" + str(idx)+ ext )
        print( img_dir)
        print( destimg_dir)

        # shutil.copyfile( img_dir, destimg_dir)
        
        # exit()
