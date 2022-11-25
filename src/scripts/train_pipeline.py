import os
from pickle import NONE
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import argparse
import warnings
from distutils.util import strtobool
import torch.nn.functional as F
import sys
import gc
from os.path import join, dirname, abspath, isfile


from utils import AverageMeter, sec_to_str, FormatInput

CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(CURRENT_DIR, '../..'))  # Import local models
from agents.PickAgent import PickAgent
warnings.filterwarnings('ignore')

# Import helper funtions
from src.dataloader.VLDataloader import VLM_dataset

def collate_fn(batch):
    return batch

def main(args):
    
    # set up device
    assert torch.cuda.is_available(), "CUDA required by CLIP model"
    device = 'cuda'
    cudnn.benchmark = True

    # load data
    train_dataset = VLM_dataset(args.data_dir, 'train', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_numbers = args.sample_numbers, train_tasks=args.train_tasks, args=args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, 
                                            pin_memory=args.pin_memory, drop_last=True, collate_fn = collate_fn, persistent_workers=True)

    val_dataset = VLM_dataset(args.data_dir, 'valid', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_numbers = args.sample_numbers, train_tasks=args.train_tasks, args=args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
                                            pin_memory=args.pin_memory, drop_last=True, collate_fn = collate_fn, persistent_workers=True)
    
    assert len(train_loader)>0, "ERROR: Empty train loader"
    assert len(val_loader)>0, "ERROR: Empty val loader"

        
    agent = PickAgent(num_rotations = 36, lr=args.lr)

    # model = TwoStreamClipLingUNetLatTransporterAgent(name="cliport_6dof",device=device, cfg=cfg, z_roll_pitch=True)
    # END REF
    # # TODO: define configurations for model
    # model_config = {}
    # # TODO: Call model
    # model = ClipPort6D(model_config, device=device)
    agent.model = agent.model.to(device)
    
    # if resuming training, load states
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint from'{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            agent.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
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
        acc_losses, acc_avg_loss = train(train_loader, agent, epoch, losses, args, timer, device)
        acc_loss_dict = {'acc_'+l_t:l.avg for l_t,l in acc_losses.items()}
        
        # evaluate agent
        val_losses, val_avg_loss = val(val_loader, agent, args, epoch, device)
        val_loss_dict = {'val_'+l_t:l.avg for l_t,l in val_losses.items()}
        
        # log to WandB
        if args.wandb:
            wandb.log({**acc_loss_dict, **val_loss_dict, 'acc_loss':acc_avg_loss, 'val_loss':val_avg_loss},step=epoch)

        # save the model
        save_name = args.checkpoint_path+'/checkpoint_{}'.format(args.checkpoint_name)
        
        if val_avg_loss<best_val_loss:
            best_val_loss = val_avg_loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': agent.model.state_dict(),
                'optimizer' : agent.optimizer.state_dict(),
                'scheduler':agent.scheduler.state_dict(),
                'train_tasks': args.train_tasks
                }, save_name + '_best'+ '.pth')
            print("New best model saved")
        if (epoch+1)%5==0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': agent.model.state_dict(),
                'optimizer' : agent.optimizer.state_dict(),
                'scheduler':agent.scheduler.state_dict(),
                'train_tasks': args.train_tasks
                }, save_name + f"_epoch{epoch}" + '.pth')
            print("Checkpoint model saved")
            if args.wandb:
                wandb.save(save_name + f"_epoch{epoch}" + '.pth')
                print("Checkpoint model saved to WandB")
                os.remove(save_name + f"_epoch{epoch}" + '.pth')

    if args.wandb: 
        wandb.save( save_name + '_best' + '.pth' )
        wandb.finish()

def train(data_loader, agent, epoch, losses, args, timer, device):
    batch_time = timer["batch_time"]
    
    loss_dict = {}
    total_loss = []
    
    end = time.time()
    for i, batch_data in enumerate(data_loader):
        inp = FormatInput(batch_data)
        if inp is None:
            print("Warning: Formatted sample is none. Skipping sample.")
            continue
        loss_dict = agent.train_agent(inp)
        
        # propogate forwards
        # loss_dict = model(inp)
        # loss_dict = {"attention_loss": torch.tensor(1.0),
        #             "xy_loss": torch.tensor(1.0),
        #             "z_loss": torch.tensor(1.0),
        #             "roll_loss": torch.tensor(1.0),
        #             "pitch_losses": torch.tensor(1.0)
        #             }

        if losses == {}:
            for loss_term in loss_dict:
                losses[loss_term] = AverageMeter(loss_term)

        for loss_term in loss_dict:
            losses[loss_term].update(loss_dict[loss_term].item(), args.batch_size)

        # update time tracking
        batch_time.update(time.time() - end)
        end = time.time()

        # log outputs at given frequency
        if i % args.log_freq == 0:
            time_left = sec_to_str((len(data_loader)-i-1) * batch_time.avg + (args.epochs-epoch-1) * batch_time.avg * len(data_loader))
            tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ' \
                        'ETA: {} '.format(
                        epoch + 1, args.epochs, i, len(data_loader), time_left, batch_time=batch_time)
            # for loss_term in losses:
            #     tmp_str += '{}: {loss.val:.4f} ({loss.avg:.4f})  '.format(loss_term, loss=losses[loss_term])
            tmp_str += 'Avg Loss: {}  '.format(torch.tensor(total_loss).mean(0, keepdim=True))
            print(tmp_str)
        
        total_loss.append(sum(l.item() for l in loss_dict.values()))
    avg_loss = torch.tensor(total_loss).mean(0, keepdim=True).to(device)

    return losses, avg_loss
   
    
def val(data_loader, agent, args, epoch, device):

    agent.model.eval()
    losses= {}
    total_loss = []

    for _, batch_data in enumerate(data_loader):

        inp = FormatInput(batch_data)
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
            losses[loss_term].update(loss_dict[loss_term].item(), args.batch_size)
        
        total_loss.append(sum(l.item() for l in loss_dict.values()))
    
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
    parser.add_argument('--data_dir', type=str, default='/dataset', help='directory of data')
    parser.add_argument('--img_size',nargs='+', type=int, default=(224, 224), help='size of dataset images (default: [224,224]])')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size for training (default: 1)')
    parser.add_argument('--workers', type=int, default=32, help='number of workers  (default: 32)')
    parser.add_argument('--preprocess', action='store_true', help="whether to use preprocess the data")
    parser.add_argument('--unused_camera_list', nargs='+', default=[None], help='list of cameras to not use (default: [None])')
    parser.add_argument('--use_fail_cases', action='store_true', help="add if use the fail cases")
    parser.add_argument('--sample_numbers', type=int, default=0, help="if greater than 0, use to limit demonstrations (Default:0)")
    parser.add_argument('--pin_memory', action='store_true', help="do not use if the RAM is small")
    parser.add_argument('--train_tasks', nargs='+', type=str, default = ["all"],
                        help="list of tasks to demonstrations (Default:all, options [all pick stack shape_sorter drop wipe pour door drawer])")
    parser.add_argument('--relative', default=False, help="whether to use related rotations to get clip port ground truth (default: False)")
    parser.add_argument('--renew_obs', default=True, help="whether to renew observations at every waypoint? (default: True)")
    parser.add_argument('--add_low_lang', default=False, help="whether to use low level descriptions in our language instructions (default: False)")
    
    #Training
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=15, type=int, help='total epochs(default: 15)')
    parser.add_argument('--log-freq', default=1, type=int, help='print log message at this many iterations (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate for Adam optimizer (default: 0.001)')
    parser.add_argument('--checkpoint_path', default='/home/mrsd/IDL/VLM/checkpoint', type=str, metavar='PATH', help='path to latest checkpoint (default: /checkpoints)')
    parser.add_argument('--checkpoint_name', default='model', type=str, metavar='NAME', help='unique name to identify the model (default: model)')
    parser.add_argument('--resume', default= None, type=str, help='use to resume training from checkpoint-path/model-best.pth')
    parser.add_argument('--wandb', action='store_true', help="whether or not to connect to WandB project")
    parser.add_argument('--wandb_entity', type=str, default="11785-vlm", help="visualize the training. Account Name")
    parser.add_argument('--wandb_project', type=str, default="11785-Final-Project",  help="visualize the training. Project Name")
    parser.add_argument('--wandb_key', type=str, default="11785-Final-Project",  help="visualize the training. Project key")

    args = parser.parse_args()
    
    main(args)