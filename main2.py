import torch, os
import torch.nn as nn
import yaml
import argparse
import numpy as np
import random
import wandb
import PIL.Image
from tqdm.auto import tqdm
from model import DDPM
from engine2 import ddpm_train, ddpm_test, generate_new_image
from utils import save_model, plot_graph, show_images, fid_graph
from data import prepare_dataloader # train_dataloader, test_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance


def main(args):
    print("\n")
    print(f"Experiment Name: {args['exp_name']}")
    print(f"Experiment Details: {args['details']}")
    print("\n")
    print(f"Dataset Name: {args['dataset_name']}")
    print(f"Seed: {args['seed']}")
    print(f"Batch Size: {args['batch_size']}")
    print(f"Number of Epochs: {args['num_epoch']}")
    print(f"Learning Rate: {args['learning_rate']}")
    print(f"WandB Project: {args['wandb_project']}")
    print(f"WandB Run Name: {args['wandb_runname']}")
    print("\n")
   

    DATASET_NAME = args['dataset_name'] # "mnist"
    SEED = args['seed'] # 64
    LEARNING_RATE = float(args['learning_rate']) # 1e-6
    NUM_EPOCHES = args['num_epoch'] # 500
    TRAIN_LOSS, TEST_LOSS = [], []
    FID_SCORE = []
    MOD_EPOCH = 10
    
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # get dataloader
    train_dataloader, test_dataloader = prepare_dataloader(args)
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fid = FrechetInceptionDistance(feature=192).to(DEVICE)

    generator = DDPM.components['unet'].to(DEVICE)

    ddpm_n_steps = DDPM.scheduler.config.num_train_timesteps
    min_beta = DDPM.scheduler.config.beta_start
    max_beta = DDPM.scheduler.config.beta_end


    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr = LEARNING_RATE)
    # optimizer = torch.optim.AdamW(generator.parameters(), lr = LEARNING_RATE)

    last_epoch = -1
    # PATH = f"check_point/LFW2_checkpoint_{last_epoch}.pth"
    # isExist = os.path.exists(PATH)

    # print(isExist)
    # if isExist:
    #     check_point = torch.load(PATH)
    #     generator.load_state_dict(check_point['model_state_dict'])
    #     optimizer.load_state_dict(check_point['optimizer_state_dict'])
    #     last_epoch = check_point['epoch'] + 1
    #     loss = check_point['loss']
    #     print("Loaded...")

    # setup wandb
    wandb.init( project=args['wandb_project'], name=args['wandb_runname'], config=args)
    
    # for epoch in tqdm(range(NUM_EPOCHES)):
    for epoch in tqdm(range(last_epoch + 1, NUM_EPOCHES)):
        
        train_loss, train_model, train_optimizer = ddpm_train(generator, train_dataloader, loss_fn, optimizer, DEVICE, ddpm_n_steps, min_beta, max_beta)
        generator = train_model
        optimizer = train_optimizer
        # test_loss = ddpm_test(DDPM, lfw_test_dataloader, loss_fn, DEVICE)

        TRAIN_LOSS.append(train_loss)
        # TEST_LOSS.append(test_loss)

        

        if (epoch+1) % MOD_EPOCH == 0 or (epoch+1) <= MOD_EPOCH:
            images = generate_new_image(generator, DEVICE, ddpm_n_steps, min_beta, max_beta, n_sample = 30)
            show_images(images, f"{epoch}", f"/scratch/data/m22cs061/DDPM/gen_images/{DATASET_NAME}/{DATASET_NAME}_{epoch+1}_{LEARNING_RATE}.jpg")
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': generator.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': train_loss
            # }, f"check_point/LFW2_checkpoint_{epoch}.pth")
            # images = (images.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.float)
            # PIL.Image.fromarray(images[0].cpu().numpy(), 'RGB').save(f'gen_images/mnist_{epoch+1}.png')
            act_images = []
            i = 0
            for batch, (image, _) in enumerate(test_dataloader):
                for im in image:
                    if i == 30:
                        break
                    act_images.append(im)
                    i += 1

            act_images = torch.stack(act_images).type(torch.uint8)
            gen_images = images.to(torch.uint8)

            fid.update(act_images.to(DEVICE), real=True)
            fid.update(gen_images.to(DEVICE), real=False)
            fid_val = fid.compute()

            FID_SCORE.append(fid_val.item())

            print(f"EPOCH [{epoch}/{NUM_EPOCHES}]: Train Loss: {train_loss:.5f}  FID: {fid_val.item():.5f}")#  Test Loss: {test_loss:.5f}")
            wandb.log({
                    "Train Loss": train_loss,
                    "FID": fid_val.item(),
                    "Generated Image": wandb.Image(f"/scratch/data/m22cs061/DDPM/gen_images/{DATASET_NAME}/{DATASET_NAME}_{epoch+1}_{LEARNING_RATE}.jpg"),
                })
            
            # checkpointing
            torch.save({
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'/scratch/data/m22cs061/DDPM/save_model/{DATASET_NAME}_checkpoint/{DATASET_NAME}_checkpoint_{epoch+1}.pth')
        


    save_model(model = generator, target_dir = "/scratch/data/m22cs061/DDPM/save_model", model_name = f"{DATASET_NAME}.pth")

    plot_graph(train_losses = TRAIN_LOSS, test_losses = TEST_LOSS,  
                fig_name = f"/scratch/data/m22cs061/DDPM/plots/{DATASET_NAME}_{LEARNING_RATE}_loss.jpg")

    fid_graph(fid_scores = FID_SCORE, fig_name = f"/scratch/data/m22cs061/DDPM/plots/{DATASET_NAME}_{LEARNING_RATE}_FID.jpg")
    
    wandb.finish()



# XIO:  fatal IO error 25 (Inappropriate ioctl for DEVICE) on X server ":87"
#       after 401 requests (401 known processed) with 2 events remaining.



if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Original Architecture of VIT")
   parser.add_argument("--config", type=str, required=True, help="Path to the config file")
   
   args = parser.parse_args()
   
   # Load config file
   with open(args.config, 'r') as file:
      config = yaml.safe_load(file)

   # Automatically generate wandb_runname
   config['wandb_runname'] = f"{config['exp_name']}_{config['dataset_name']}"
   
   main(config)