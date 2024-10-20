import torch, os
import torch.nn as nn
import PIL.Image
from tqdm.auto import tqdm
from model import DDPM
from engine import ddpm_train, ddpm_test, generate_new_image
from utils import save_model, plot_graph, show_images, fid_graph
from data import lfw_train_dataloader, lfw_test_dataloader
from torchmetrics.image.fid import FrechetInceptionDistance


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
fid = FrechetInceptionDistance(feature=192).to(device)

LEARNING_RATE = 2e-4
NUM_EPOCHES = 200
TRAIN_LOSS, TEST_LOSS = [], []
FID_SCORE = []


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(DDPM.components['unet'].parameters(), lr = LEARNING_RATE)

last_epoch = 0
PATH = f"check_point/LFW_checkpoint_{last_epoch}.pth"
isExist = os.path.exists(PATH)

print(isExist)
if isExist:
    check_point = torch.load(PATH)
    DDPM.components['unet'].load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    last_epoch = check_point['epoch'] + 1
    loss = check_point['loss']
    print("Loaded...")


# for epoch in tqdm(range(NUM_EPOCHES)):
for epoch in tqdm(range(last_epoch + 1, NUM_EPOCHES)):
    
    train_loss, train_model = ddpm_train(DDPM, lfw_train_dataloader, loss_fn, optimizer, device)
    DDPM = train_model
    # test_loss = ddpm_test(DDPM, lfw_test_dataloader, loss_fn, device)

    TRAIN_LOSS.append(train_loss)
    # TEST_LOSS.append(test_loss)

    

    if (epoch+1) % 10 == 0 or (epoch+1) <= 10:
        images = generate_new_image(DDPM, device, n_sample = 30).to(torch.uint8)
        show_images(images, f"{epoch}", f"gen_images/lfw_{epoch}.jpg")
        torch.save({
            'epoch': epoch,
            'model_state_dict': DDPM.components['unet'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }, f"check_point/LFW_checkpoint_{epoch}.pth")
        # images = (images.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.float)
        # PIL.Image.fromarray(images[0].cpu().numpy(), 'RGB').save(f'gen_images/mnist_{epoch+1}.png')
        act_images = []
        i = 0
        for batch, (image, _) in enumerate(lfw_test_dataloader):
            for im in image:
                if i == 30:
                    break
                act_images.append(im)
                i += 1

        act_images = torch.stack(act_images).type(torch.uint8)
        gen_images = images.to(torch.uint8)

        fid.update(act_images.to(device), real=True)
        fid.update(gen_images.to(device), real=False)
        fid_val = fid.compute()

        FID_SCORE.append(fid_val.item())

        print(f"EPOCH [{epoch}/{NUM_EPOCHES}]: Train Loss: {train_loss:.5f}  FID: {fid_val.item():.5f}")#  Test Loss: {test_loss:.5f}")
    


save_model(model = DDPM, target_dir = "./save_model", model_name = f"la_lfw_DDPM_UNET_Face.pth")

plot_graph(train_losses = TRAIN_LOSS, test_losses = TEST_LOSS,  
            fig_name = f"plots/la_lfw_train_Loss_and_accuracy_plot_DDPM_Face.jpg")

fid_graph(fid_scores = FID_SCORE, fig_name = f"plots/la_lfw_DDPM_FID.jpg")



# XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server ":87"
#       after 401 requests (401 known processed) with 2 events remaining.