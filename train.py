import argparse
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from datetime import datetime

import train_config as config
from model import HanelsoAlexNetModel
from custom_dataset import HanelsoAlexnetDataset

def create_dataloaders( batch_size: int, val_split: float ) -> tuple[ DataLoader, DataLoader ]:
    data_dir = config.train_data_dir
    transform = config.transform
    shuffle = config.shuffle

    dataset = HanelsoAlexnetDataset( data_path = data_dir, transform = transform )
    val_split_cnt = int( len(dataset) * val_split )
    train_split_cnt = len( dataset ) - val_split_cnt
    train_set, val_set = random_split( dataset, [ train_split_cnt, val_split_cnt ] )

    train_loader = DataLoader( train_set, batch_size = batch_size, shuffle = shuffle )
    val_loader = DataLoader( val_set, batch_size = batch_size, shuffle = shuffle )

    return train_loader, val_loader

def validate( 
            model: HanelsoAlexNetModel,
            val_loader: DataLoader,
            criterion: nn.modules.loss._Loss
    ) -> float:
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        with tqdm( val_loader, desc = f"Validating:", unit = "batch" ) as tbar:
            for batch in tbar:
                images, labels = batch
                out = model( images.to( config.device ) )
                loss = criterion( out, labels.to( config.device ) )
                val_loss += loss.item()
        avg_val_loss = val_loss / len( val_loader )

    return avg_val_loss
    

def run_epoch( 
            model: HanelsoAlexNetModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epoch: int,
            optimizer: optim.Optimizer,
            criterion: nn.modules.loss._Loss
    ):
    model.train()
    running_loss = 0
    
    with tqdm( train_loader, desc = f"Epoch { epoch + 1 }", unit = "batch" ) as tbar:
        for batch in tbar:
            images, labels = batch
            out = model( images.to( config.device ))
            loss = criterion( out, labels.to( config.device ) )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    average_loss = running_loss / len( train_loader )
    average_val_loss = validate( model = model, val_loader = val_loader, criterion = criterion )
    print( f"Training loss: {average_loss} | Validation loss: {average_val_loss}" )
    

def run():

    parser = argparse.ArgumentParser( description = "Hanelso AlexNet Model Train" )
    parser.add_argument( "--epochs", type = int, default = config.epochs )
    parser.add_argument( "--batch_size", type = int, default = config.batch_size )
    parser.add_argument( "--lr", type = int, default = config.learning_rate )
    parser.add_argument( "--val_split", type = float, default = config.val_split_percent )
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    val_split_percent = args.val_split

    n_classes = config.n_classes
    device = config.device
    loss_label_smoothing = config.loss_label_smoothing

    model = HanelsoAlexNetModel( n_classes= n_classes ).to( device )
    train_loader, val_loader = create_dataloaders( batch_size = batch_size, val_split = val_split_percent )
    optimizer = optim.Adam( model.parameters(), lr = learning_rate )
    criterion = nn.CrossEntropyLoss( label_smoothing= loss_label_smoothing ).to( device )
    current_datetime_str = datetime.now().strftime( "%Y%m%d_%H%M%S" )
    
    for epoch in range( epochs ):
        run_epoch( 
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            epoch = epoch,
            optimizer = optimizer,
            criterion = criterion
        )

        torch.save( model.state_dict(), f"model_{ current_datetime_str }_epoch{ epoch + 1 }.pt" )
    

if __name__ == "__main__":
    run()

