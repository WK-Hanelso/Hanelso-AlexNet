import os
import json

import config_train as config

class_index_name = "class_index.json"
data_path = config.train_data_dir

tmp_list = sorted(os.listdir( os.path.join(data_path, "images" )))
# print( tmp_list )

tmp_dict = dict()
'''
{
    class index : cls_name
}
'''
for idx, cls_name in enumerate( tmp_list ):
    # print( idx, cls_name )
    tmp_dict[idx] = cls_name

for key, value in tmp_dict.items():
    print( f"key : { key }, value : { value }")

# with open( os.path.join( data_path, class_index_name ), 'w' ) as f:
#     json.dump(tmp_dict, f )