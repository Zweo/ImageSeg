import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import utils
from skimage import color
from losses import CombinedLoss


def create_exp_directory(exp_dir_name):
    if not os.path.exists(exp_dir_name):
        try:
            os.makedirs(exp_dir_name)
            print("Successfully Created Directory @ {}".format(exp_dir_name))
        except Exception:
            print("Directory Creation Failed - Check Path")
    else:
        print("Directory {} Exists ".format(exp_dir_name))


def plot_predictions(images_batch, labels_batch, batch_output, plt_title,
                     file_save_name):
    f = plt.figure(figsize=(20, 20))
    n, c, h, w = images_batch.shape
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    grid = utils.make_grid(images_batch.cpu(), nrow=4)
    plt.subplot(131)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Slices')
    grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(132)
    plt.imshow(color_grid)
    plt.title('Ground Truth')
    grid = utils.make_grid(batch_output.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(133)
    plt.imshow(color_grid)
    plt.title('Prediction')

    plt.suptitle(plt_title)
    plt.tight_layout()

    f.savefig(file_save_name, bbox_inches='tight')
    plt.close(f)
    plt.gcf().clear()


class Solver(object):
    def __init__(self, num_classes, optimizer, optimizer_args={}):
        self.lr_scheduler_args = {"gamma": 0.5, "step_size": 8}
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer
        self.loss_func = CombinedLoss(weight_dice=0, weight_ce=100)
        self.num_classes = num_classes
        self.classes = list(range(self.num_classes))

    def train(self,
              model,
              train_loader,
              num_epochs,
              log_params,
              expdir,
              resume=True):
        create_exp_directory(expdir)
        create_exp_directory(log_params["logdir"])
        optimizer = self.optimizer(model.parameters(), **self.optimizer_args)
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_scheduler_args["step_size"],
            gamma=self.lr_scheduler_args["gamma"])
        epoch = -1
        print('-------> Starting to train')
        if resume:
            try:
                prior_model_paths = sorted(glob.glob(
                    os.path.join(expdir, 'Epoch_*')),
                                           key=os.path.getmtime)
                current_model = prior_model_paths.pop()
                state = torch.load(current_model)
                model.load_state_dict(state["model_state_dict"])
                epoch = state["epoch"]
                print("Successfully Resuming from Epoch {}".format(epoch + 1))
            except Exception as e:
                print("No model to restore. {}".format(e))
        log_params["logger"].info("{} parameters in total".format(
            sum(x.numel() for x in model.parameters())))

        model.train()
        while epoch < num_epochs:
            epoch = epoch + 1
            epoch_start = time.time()
            loss_batch = np.zeros(1)
            loss_dice_batch = np.zeros(1)
            loss_ce_batch = np.zeros(1)
            for batch_idx, sample in enumerate(train_loader):
                images, labels = sample
                images, labels = Variable(images), Variable(labels)
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                predictions = model(images)
                loss_total, loss_dice, loss_ce = self.loss_func(
                    inputx=predictions, target=labels)
                loss_total.backward()
                optimizer.step()

                loss_batch += loss_total.item()
                loss_dice_batch += loss_dice.item()
                loss_ce_batch += loss_ce.item()
                _, batch_output = torch.max(predictions, dim=1)
                if batch_idx == len(train_loader) - 2:
                    plt_title = 'Trian Results Epoch ' + str(epoch)
                    file_save_name = os.path.join(
                        log_params["logdir"],
                        'Epoch_{}_Trian_Predictions.pdf'.format(epoch))
                    plot_predictions(images, labels, batch_output, plt_title,
                                     file_save_name)

                if batch_idx % (len(train_loader) // 2) == 0:
                    log_params["logger"].info(
                        "Epoch: {} lr:{} [{}/{}] ({:.0f}%)]"
                        "with loss: {},\ndice_loss:{},ce_loss:{}".format(
                            epoch, optimizer.param_groups[0]['lr'], batch_idx,
                            len(train_loader),
                            100. * batch_idx / len(train_loader),
                            loss_batch / (batch_idx + 1),
                            loss_dice_batch / (batch_idx + 1),
                            loss_ce_batch / (batch_idx + 1)))

            scheduler.step()
            epoch_finish = time.time() - epoch_start
            log_params["logger"].info(
                "Train Epoch {} finished in {:.04f} seconds.".format(
                    epoch, epoch_finish))

            # Saving Models
            if epoch % log_params["log_iter"] == 0:  # 每log_iter次保存一次模型
                save_name = os.path.join(
                    expdir,
                    'Epoch_' + str(epoch).zfill(2) + '_training_state.pkl')
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch
                }
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                torch.save(checkpoint, save_name)
