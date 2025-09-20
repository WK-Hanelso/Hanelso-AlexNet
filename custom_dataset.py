import os
import json

import torch
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class HanelsoAlexnetDataset( Dataset ):

    def __init__( self, data_path: str, transform: transforms.Compose | None = None ):

        self.data_path = data_path
        self.image_paths = list()
        self.labels = list()
        self.class_mapping = dict()

        class_index_path = os.path.join( data_path, "class_index.json" )

        with open( class_index_path, 'r' ) as f:
            class_id_to_name = json.load( f )
        # class_id_to_name = { name[0] : [cls_id, name[1]] for cls_id, name in class_id_to_name.items() }
        class_id_to_name = { name : [cls_id, name] for cls_id, name in class_id_to_name.items() }
        print( class_id_to_name)

        image_dir = os.path.join( data_path, "images" )
        for class_name in sorted( os.listdir( image_dir ) ):
            class_path = os.path.join( image_dir, class_name )
            for image_name in sorted( os.listdir( class_path ) ):
                image_path = os.path.join( class_path, image_name )
                self.image_paths.append( image_path )

                class_map = class_id_to_name[ class_name ]
                self.class_mapping[ int(class_map[0]) ] = class_map[1]
                self.labels.append( int(class_map[0]) )

        self.transform = transform

    def __len__( self ) -> int:
        return len( self.image_paths )

    def __getitem__( self, idx: int ) -> tuple[ torch.Tensor, torch.Tensor ]:
        image_path = self.image_paths[ idx ]
        label = torch.tensor( self.labels[ idx ] )

        # Load image as PIL format
        image = Image.open( image_path ).convert( "RGB" )   # Ensure RGB mode

        if self.transform:
            image = self.transform( image )

        return image, label
        