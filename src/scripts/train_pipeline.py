import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import warnings
import sys
import gc
from os.path import join, dirname, abspath
import traceback

from utils import AverageMeter, sec_to_str

CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(CURRENT_DIR, '../..'))  # Import local models
from agents.PickAgent import PickAgent
warnings.filterwarnings('ignore')

# Import helper funtions
from cliport.dataset import RavensDataset

def main(args):
    
    # set up device
    assert torch.cuda.is_available(), "CUDA required by CLIP model"
    device = 'cuda'
    cudnn.benchmark = True

    dataset_cfg = {"dataset":{"type": "single",
                "images": True,
                "cache": True,
                "augment":{"theta_sigma":60}}}

    # load data
    train_dataset = RavensDataset(args.train_data_dir, dataset_cfg, n_demos=1, augment=True)

    val_dataset = RavensDataset(args.val_data_dir, dataset_cfg, n_demos=1, augment=True)
    
    assert len(train_dataset)>0, "ERROR: Empty train dataset"
    assert len(val_dataset)>0, "ERROR: Empty val dataset"

    agent = PickAgent(num_rotations = 1, lr=args.lr)
    agent.model = agent.model.to(device)
    
    # if resuming training, load states
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint from'{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            agent.model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    
    # set up WandB
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.checkpoint_name, group=args.train_tasks[0], config=args )
        wandb.config.update(args)

    # initialize loss and time tracking
    losses = {}
    timer = {"batch_time":AverageMeter('Time', ':6.3f')}

    # start training
    best_val_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        acc_losses, acc_avg_loss = train(train_dataset, agent, epoch, losses, args, timer, device)
        acc_loss_dict = {'acc_'+l_t:l.avg for l_t,l in acc_losses.items()}
        
        # evaluate agent
        val_losses, val_avg_loss = val(val_dataset, agent, args, epoch, device)
        val_loss_dict = {'val_'+l_t:l.avg for l_t,l in val_losses.items()}
        
        # log to WandB
        if args.wandb:
            # wandb.log({**acc_loss_dict, **val_loss_dict, 'acc_loss':acc_avg_loss, 'val_loss':val_avg_loss},step=epoch)
            try:
                wandb.log({**acc_loss_dict, **val_loss_dict, 'acc_loss':acc_avg_loss, 'val_loss':val_avg_loss},step=epoch)
            except Exception as e:
                print(traceback.print_exc(), file=sys.stderr)

        # save the model
        save_name = args.checkpoint_path+'/checkpoint_{}'.format(args.checkpoint_name)
        
        if val_avg_loss<best_val_loss:
            best_val_loss = val_avg_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': agent.model.state_dict(),
                'optimizer' : agent.optimizer.state_dict(),
                # 'scheduler':agent.scheduler.state_dict(),
                'train_tasks': args.train_tasks
                }, save_name + '_best'+ '.pth')
            print("New best model saved")
        if (epoch+1)%5==0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': agent.model.state_dict(),
                'optimizer' : agent.optimizer.state_dict(),
                # 'scheduler':agent.scheduler.state_dict(),
                'train_tasks': args.train_tasks
                }, save_name + f"_epoch{epoch}" + '.pth')
            print("Checkpoint model saved")
            if args.wandb:
                try:
                    wandb.save(save_name + f"_epoch{epoch}" + '.pth')
                    print("Checkpoint model saved to WandB")

                except Exception as e:
                    print(traceback.print_exc(), file=sys.stderr)
            
            os.remove(save_name + f"_epoch{epoch}" + '.pth')
            

    if args.wandb: 
        try:
            wandb.save( save_name + '_best' + '.pth' )
            wandb.finish()
        except Exception as e:
            print(traceback.print_exc(), file=sys.stderr)

def train(data_loader, agent, epoch, losses, args, timer, device):
    batch_time = timer["batch_time"]
    
    loss_dict = {}
    total_loss = []
    
    end = time.time()
    for i in range(100):
        batch_data = next(iter(data_loader))
        inp, _ = batch_data

        loss_dict['attention_loss'] = agent.train_agent(inp)

        if losses == {}:
            for loss_term in loss_dict:
                losses[loss_term] = AverageMeter(loss_term)

        for loss_term in loss_dict:
            losses[loss_term].update(loss_dict[loss_term].item(), 1)

        # update time tracking
        batch_time.update(time.time() - end)
        end = time.time()

        # Compute total loss
        total_loss.append(sum(l.item() for l in loss_dict.values()))

        # log outputs at given frequency
        if i % args.log_freq == 0:
            time_left = sec_to_str((100-i-1) * batch_time.avg + (args.epochs-epoch-1) * batch_time.avg * 100)
            tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                        'ETA: {} '.format(
                        epoch + 1, args.epochs, i, 100, time_left, batch_time=batch_time)
            # for loss_term in losses:
            #     tmp_str += '{}: {loss.val:.4f} ({loss.avg:.4f})  '.format(loss_term, loss=losses[loss_term])
            tmp_str += 'Avg Loss: {}  '.format(torch.tensor(total_loss).mean(0, keepdim=True))
            print(tmp_str)
        
    avg_loss = torch.tensor(total_loss).mean(0, keepdim=True).to(device)

    return losses, avg_loss
   
    
def val(data_loader, agent, args, epoch, device):

    agent.model.eval()
    losses= {}
    total_loss = []

    for i in range(100):
        batch_data = next(iter(data_loader))
        inp, _ = batch_data
        if inp is None: pass

        with torch.no_grad():
            # loss_dict = model(inp)
            loss_dict = {"attention_loss": torch.tensor(1),
                    "xy_loss": torch.tensor(1),
                    "z_loss": torch.tensor(1),
                    "roll_loss": torch.tensor(1),
                    "pitch_losses": torch.tensor(1)
                    }

        if losses == {}:
            for loss_term in loss_dict:
                losses[loss_term] = AverageMeter(loss_term)

        for loss_term in loss_dict:
            losses[loss_term].update(loss_dict[loss_term].item(), 1)
        
        total_loss.append(sum(l.item() for l in loss_dict.values()))
        break
    
    avg_loss = torch.mean(torch.FloatTensor(total_loss), dim=0, keepdim=True).to(device)

    tmp_str = 'Epoch [{}/{}] Val_loss: {} '.format(epoch + 1, args.epochs, avg_loss)
    # for loss_term in losses:
    #     tmp_str += '{}: {loss.val} ({loss.avg})  '.format(loss_term, loss=losses[loss_term])
    # print(tmp_str)

    return losses, avg_loss

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    ## Maintained most arguments from VLMbench
    #Dataset processing
    #add num rot
    parser.add_argument('--train_data_dir', type=str, default='/home/ubuntu/cliport/data/stack-block-pyramid-seq-seen-colors-test', help='directory of training data')
    parser.add_argument('--val_data_dir', type=str, default='/home/ubuntu/cliport/data/stack-block-pyramid-seq-seen-colors-test', help='directory of validation data')
    parser.add_argument('--train_tasks', nargs='+', type=str, default = ["pick"],
                        help="list of tasks to demonstrations (Default:all, options [all pick stack shape_sorter drop wipe pour door drawer])")
    
    #Training
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=2000, type=int, help='total epochs(default: 15)')
    parser.add_argument('--log-freq', default=10, type=int, help='print log message at this many iterations (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--checkpoint_path', default='/home/ubuntu/VLM/checkpoint', type=str, metavar='PATH', help='path to latest checkpoint (default: /checkpoints)')
    parser.add_argument('--checkpoint_name', default='model', type=str, metavar='NAME', help='unique name to identify the model (default: model)')
    parser.add_argument('--resume', default= None, type=str, help='use to resume training from checkpoint-path/model-best.pth')
    parser.add_argument('--wandb', action='store_true', help="whether or not to connect to WandB project")
    parser.add_argument('--wandb_entity', type=str, default="11785-vlm", help="visualize the training. Account Name")
    parser.add_argument('--wandb_project', type=str, default="11785-Final-Project",  help="visualize the training. Project Name")
    parser.add_argument('--wandb_key', type=str, default="11785-Final-Project",  help="visualize the training. Project key")

    args = parser.parse_args()
    
    main(args)