import os
import time
import numpy as np
import torch
import argparse
import warnings
import sys
import gc
from os.path import join, dirname, abspath
import traceback

from utils import AverageMeter, sec_to_str

CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(CURRENT_DIR, '../..'))  # Import local models
from agents.PickPlaceAgent import PickPlaceAgent
warnings.filterwarnings('ignore')

# Import helper funtions
from cliport.dataset import RavensDataset

def main(args):
    
    # set up device
    assert torch.cuda.is_available(), "CUDA required by CLIP model"
    device = 'cuda'

    train_dataset_cfg = {"dataset":{"type": "single",
                        "images": True,
                        "cache": True,
                        "augment":{"theta_sigma":60},
                        "cache_size": 350},
                        }

    # load data
    train_dataset = RavensDataset(args.train_data_dir, train_dataset_cfg, n_demos=1000, augment=True)

    val_dataset_cfg = {"dataset":{"type": "single",
                    "images": True,
                    "cache": False,
                    "augment":{"theta_sigma":60},
                    "cache_size": 0},
                    }

    val_dataset = RavensDataset(args.val_data_dir, val_dataset_cfg, n_demos=100, augment=False)

    agent = PickPlaceAgent(num_rotations=args.num_rotations, lr=args.lr, device=device)
    
    # if resuming training, load states
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint from'{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            agent.pick_model.load_state_dict(checkpoint['pick_state_dict'])
            agent.place_model.load_state_dict(checkpoint['place_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    
    # set up WandB
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.checkpoint_name, config=args )

    # initialize loss and time tracking
    timer = {"batch_time": AverageMeter('Time', ':6.3f')}

    val_losses = eval(val_dataset, agent, args)

    # start training
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train_losses = train(train_dataset, agent, epoch, args, timer)
        train_losses = {'train_'+l_t:l.avg for l_t,l in train_losses.items()}
        
        # evaluate agent
        val_losses = eval(val_dataset, agent, args)
        val_losses = {'val_'+l_t:l.avg for l_t,l in val_losses.items()}
        
        # log to WandB
        if args.wandb:
            try:
                # TODO: Handle reinitialization
                wandb.log({**train_losses, **val_losses, 
                            "lr":float(agent.optimizer.param_groups[0]['lr'])})
            except Exception as e:
                print(traceback.print_exc(), file=sys.stderr)

        # save the model
        save_name = args.checkpoint_path+'/checkpoint_{}'.format(args.checkpoint_name)

        torch.save({
                'epoch': epoch + 1,
                'pick_state_dict': agent.pick_model.state_dict(),
                'place_state_dict': agent.place_model.state_dict(),
                'optimizer' : agent.optimizer.state_dict(),
                }, save_name + '_latest'+ '.pth')

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'pick_state_dict': agent.pick_model.state_dict(),
                'place_state_dict': agent.place_model.state_dict(),
                'optimizer' : agent.optimizer.state_dict(),
                }, save_name + f"_epoch{epoch}" + '.pth')
            print("Checkpoint model saved")
            if args.wandb:
                try:
                    wandb.save(save_name + f"_epoch{epoch}" + '.pth')
                    print("Checkpoint model saved to WandB")
                except Exception as e:
                    print(traceback.print_exc(), file=sys.stderr)            

    # if args.wandb: 
    #     try:
    #         wandb.save( save_name + '_best' + '.pth' )
    #         wandb.finish()
    #     except Exception as e:
    #         print(traceback.print_exc(), file=sys.stderr)

def train(data_loader, agent, epoch, args, timer):
    batch_time = timer["batch_time"]
    
    average_losses = {}
    
    end = time.time()
    for i in range(args.itrs_per_epoch):
        batch_data = next(iter(data_loader))
        inp, _ = batch_data

        loss_dict = agent.train_agent(inp)

        if average_losses == {}:
            for loss_term in loss_dict:
                average_losses[loss_term] = AverageMeter(loss_term)

        for loss_term in loss_dict:
            average_losses[loss_term].update(loss_dict[loss_term], 1)

        # update time tracking
        batch_time.update(time.time() - end)
        end = time.time()

        # log outputs at given frequency
        if i % args.log_freq == 0:
            time_left = sec_to_str((args.itrs_per_epoch-i-1) * batch_time.avg + (args.epochs-epoch-1) * batch_time.avg * args.itrs_per_epoch)
            tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                        'ETA: {}'.format(
                        epoch + 1, args.epochs, i, args.itrs_per_epoch, time_left, batch_time=batch_time)
            for key, avg_meter in average_losses.items():
                tmp_str += " {}: {}".format(key, avg_meter.avg)
            print(tmp_str)
        
    return average_losses
   
    
def eval(data_loader, agent, args):
    average_losses = {}

    for _ in range(args.eval_itrs):
        batch_data = next(iter(data_loader))
        inp, _ = batch_data

        loss_dict = agent.eval_agent(inp)

        if average_losses == {}:
            for loss_term in loss_dict:
                average_losses[loss_term] = AverageMeter(loss_term)

        for loss_term in loss_dict:
            average_losses[loss_term].update(loss_dict[loss_term], 1)
    
    tmp_str = "Validation:"
    for key, avg_meter in average_losses.items():
        tmp_str += " {}: {}".format(key, avg_meter.avg)
    print(tmp_str)

    return average_losses

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    ## Maintained most arguments from VLMbench
    #Dataset processing
    #add num rot
    parser.add_argument('--train_data_dir', type=str, default='/home/ubuntu/cliport/data/packing-boxes-pairs-full-train', help='directory of training data')
    parser.add_argument('--val_data_dir', type=str, default='/home/ubuntu/cliport/data/packing-boxes-pairs-full-val', help='directory of validation data')
    
    #Training
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--itrs_per_epoch', default=100, type=int, help='Iteratins per epoch(default: 100)')
    parser.add_argument('--epochs', default=2000, type=int, help='total epochs(default: 2000)')
    parser.add_argument('--eval_itrs', default=100, type=int, help='Number of steps every time you run eval(default: 100)')
    parser.add_argument('--log-freq', default=10, type=int, help='print log message at this many iterations (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--checkpoint_path', default='/home/ubuntu/VLM/checkpoints', type=str, metavar='PATH', help='path to latest checkpoint (default: /checkpoints)')
    parser.add_argument('--checkpoint_name', default='model', type=str, metavar='NAME', help='unique name to identify the model (default: model)')
    parser.add_argument('--resume', default= None, type=str, help='use to resume training from checkpoint-path/model-best.pth')
    parser.add_argument('--wandb', action='store_true', help="whether or not to connect to WandB project")
    parser.add_argument('--wandb_entity', type=str, default="11785-vlm", help="visualize the training. Account Name")
    parser.add_argument('--wandb_project', type=str, default="11785-Final-Project",  help="visualize the training. Project Name")
    parser.add_argument('--num_rotations', type=int, default="12",  help="number of rotations if model is place (pick always uses 1 rotation)")

    args = parser.parse_args()
    
    main(args)