import os
import argparse
import torch
import torch.nn as nn

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from datetime import datetime

import config_train as config
from model import HanelsoAlexNetModel
from custom_dataset import HanelsoAlexnetDataset

log_file_name = "log.txt"

def create_dataloaders( batch_size: int, val_split: float ) -> tuple[ DataLoader, DataLoader ]:
    train_data_dir = config.train_data_dir
    val_data_dir = config.val_data_dir
    shuffle = config.shuffle
    num_workers = config.num_workers

    dataset_train = HanelsoAlexnetDataset( data_path = train_data_dir, transform = config.transform_train, filelist="train.txt" )
    dataset_val = HanelsoAlexnetDataset( data_path = val_data_dir, transform = config.transform_eval, filelist="val.txt" )

    train_loader = DataLoader( 
                        dataset_train, 
                        batch_size = batch_size, 
                        shuffle = shuffle,
                        num_workers = num_workers,
                        pin_memory = torch.cuda.is_available(),
                    )

    val_loader = DataLoader( 
                        dataset_val, 
                        batch_size = batch_size, 
                        shuffle = False,
                        num_workers = num_workers,
                        pin_memory = torch.cuda.is_available(),
                    )

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
        for idx, batch in enumerate(tbar):
            images, labels = batch

            # 처음 3배치만 통계 출력
            if idx < 3:
                print("img mean/std:", images.mean().item(), images.std().item())

            out = model( images.to( config.device ))

            if idx < 3:
                print("logits mean/std:",
                    out.detach().mean().item(),
                    out.detach().std().item())

            loss = criterion( out, labels.to( config.device ) )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    average_loss = running_loss / len( train_loader )
    average_val_loss = validate( model = model, val_loader = val_loader, criterion = criterion )

    loss_str = f"epoch: {epoch+1} || Training loss: {average_loss} | Validation loss: {average_val_loss}\n"

    print( loss_str )
    return loss_str

    

def run():

    parser = argparse.ArgumentParser( description = "Hanelso AlexNet Model Train" )
    parser.add_argument( "--epochs", type = int, default = config.epochs )
    parser.add_argument( "--batch_size", type = int, default = config.batch_size )
    parser.add_argument( "--lr", type = float, default = config.learning_rate )
    parser.add_argument( "--val_split", type = float, default = config.val_split_percent )
    parser.add_argument( "--save_dir", type = str, default = config.save_dir )
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    val_split_percent = args.val_split

    n_classes = config.n_classes
    device = config.device
    loss_label_smoothing = config.loss_label_smoothing
    save_dir = config.save_dir

    model = HanelsoAlexNetModel( n_classes= n_classes ).to( device )
    train_loader, val_loader = create_dataloaders( batch_size = batch_size, val_split = val_split_percent )
    optimizer = optim.Adam( model.parameters(), lr = learning_rate )
    criterion = nn.CrossEntropyLoss( label_smoothing= loss_label_smoothing ).to( device )
    current_datetime_str = datetime.now().strftime( "%Y%m%d_%H%M%S" )

    if os.path.exists( save_dir ) == False:
        os.mkdir( save_dir )
        
    for epoch in range( epochs ):
        
        log_str = run_epoch( 
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            epoch = epoch,
            optimizer = optimizer,
            criterion = criterion
        )
        
        save_path = os.path.join( save_dir, f"model_{ current_datetime_str }_epoch{ epoch + 1 }.pt" )
        log_path = os.path.join( save_dir, log_file_name )

        if epoch % 10 == 0:
            torch.save( model.state_dict(), save_path )

        with open( log_path, 'a' ) as f:
            f.write( log_str )
    

if __name__ == "__main__":
    run()

