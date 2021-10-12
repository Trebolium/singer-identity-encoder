from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_data import partials_n_frames
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from pathlib import Path
import torch, os, pdb
from torch import nn
from encoder.tester import collater
from torch.utils.tensorboard import SummaryWriter

def sync(device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

class SingerIdentityEncoder:
    def __init__(self, config) -> None:
        """Initialise configurations"""

        self.config = config
        # Setup the device on which to run the forward pass and the loss. These can be different, 
        # because the forward pass is faster on the GPU whereas the loss is often (depending on your
        # hyperparameters) faster on the CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FIXME: currently, the gradient is None if loss_device is cuda
        self.loss_device = torch.device("cpu")

        # Create a dataset and a dataloader
        self.dataset = SpeakerVerificationDataset(config.clean_data_root)
        self.loader = SpeakerVerificationDataLoader(
            self.dataset,
            speakers_per_batch,
            utterances_per_speaker,
            num_workers=8,
        )
        # Configure file path for the model
        self.save_path = os.path.join(config.this_model_dir, config.run_id + ".pt")
        self.backup_dir = os.path.join(config.this_model_dir, "backups")
        # Create the model and the optimizer
        self.model = SpeakerEncoder(self.device, self.loss_device, self.dataset.num_speakers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate_init)
        self.writer = self.config_model(config, self.save_path)
        self.current_step = 1
        self.losses = {'ge2e':None, 'pred':None, 'both':None}
        self.mode_iters = {'train':self.config.train_iters, 'val':self.config.val_iters}
        self.print_freq = 10

    def train(self):
        training_complete = False
        while training_complete == False:
            mode = 'train'
            self.model.train()
            self.batch_iterate(mode)
            mode = 'val'
            self.model.eval()
            with torch.no_grad():
                self.batch_iterate(mode)
            self.current_step += self.mode_iters['train']
            if self.current_step >= self.config.stop_at_step:
                training_complete = True
        print('Training complete')

    def batch_iterate(self, mode):

        # Training loop infinite as loader is infinite
        print(f'---{mode.upper()}---')
        if mode == 'train': start = self.current_step
        else: start = 1
        for step, speaker_batch in enumerate(self.loader, start):
            
            x_data_npy, y_data_npy = speaker_batch.data[0], speaker_batch.data[1]
            # Forward pass
            inputs = torch.from_numpy(x_data_npy).to(self.device).float() # speakerbatch shape = speakers, timesteps, features
            y_data = torch.from_numpy(y_data_npy).to(self.device)
            sync(self.device)
            embeds, predictions = self.model(inputs)
            sync(self.device)
            # reshape output to reflect: speaker, uttrs, embeddings
            ge2e_loss, pred_loss = self.get_losses(embeds, predictions, y_data)
            self.losses['ge2e'] = ge2e_loss
            self.losses['pred'] = pred_loss
            self.losses['both'] = ge2e_loss+pred_loss
            accuracy = self.get_accuracy(predictions, y_data)
            self.backprop_ops(ge2e_loss, pred_loss, accuracy, mode, step)
            self.periodic_ops(step)
            
            """Monitor by printing"""
            if step % self.print_freq == 0:
                if mode == 'train': print(f'Steps {step}/{self.config.stop_at_step}, Accuracy: {round(accuracy, 4)}, GE2E loss: {round(ge2e_loss.item(), 4)}, Pred loss: {round(pred_loss.item(), 4)}')
                else: print(f'Steps {step}/{self.mode_iters[mode]}, Accuracy: {round(accuracy, 4)}, GE2E loss: {round(ge2e_loss.item(), 4)}, Pred loss: {round(pred_loss.item(), 4)}') 

            """When desired number of training steps reached"""

            if step >= (start + self.mode_iters[mode]):
                break

    def config_model(self, config, save_path):
        # Load any existing model
        if config.run_id == 'testrun':
            # delete backups and save_paths
            writer = SummaryWriter('testRuns/test')
            print("Default model. Saving progress to testruns directory")
        else:
            if not config.force_restart:
                if os.path.exists(save_path):
                    checkpoint = torch.load(save_path)
                    self.current_step = checkpoint["step"]
                    self.model.load_state_dict(checkpoint["model_state"])
                    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                    self.optimizer.param_groups[0]["lr"] = learning_rate_init
                    writer = SummaryWriter(comment = '_' +config.run_id)
                    print("Model \"%s\" found, resuming training." % config.run_id)
                else:
                    writer = SummaryWriter(comment = '_' +config.run_id)
                    if not os.path.exists(config.this_model_dir):
                        os.mkdir(config.this_model_dir)
                    open(config.this_model_dir +'/config.txt', 'a').write(config.string_sum)
                    print("Starting training from scratch.")
            else:
                print('Force restart enabled!')
                writer = SummaryWriter(comment = '_' +config.run_id)
                if not os.path.exists(config.this_model_dir):
                    os.mkdir(config.this_model_dir)
                open(config.this_model_dir +'/config.txt', 'a').write(config.string_sum)
                print("Starting training from scratch.")
        return writer

    def get_losses(self, embeds, predictions, y_data):
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(self.loss_device)
        # this is the loss for one pass, reflecting 64 speakers of 10 utterances
        ge2e_loss, eer = self.model.loss(embeds_loss)
        pred_loss = nn.functional.cross_entropy(predictions, y_data)
        return ge2e_loss, pred_loss

    def get_accuracy(self, predictions, y_data):
        _, predicted = torch.max(predictions.data, 1)
        correct_preds = (predicted == y_data).sum().item()
        sync(self.loss_device)
        return correct_preds / len(y_data) #accuracy

    def backprop_ops(self, ge2e_loss, pred_loss, accuracy, mode, step):
        # Backward pass
        if mode == 'train':
            self.losses[self.config.use_loss].backward() #if loss was generated without using parameters adjusted in do_gradient_ops(), there will be no gradient after backwards upon which to adjust
            self.model.do_gradient_ops(self.losses[self.config.use_loss])
            self.optimizer.step()
            self.model.zero_grad()
        self.writer.add_scalar(f'Ge2e Loss/{mode}', ge2e_loss, step)
        self.writer.add_scalar(f'Classification Loss/{mode}', pred_loss, step)
        self.writer.add_scalar(f'Accuracy/{mode}', accuracy)

        """No need to DISPLAY combined loss anymore, but still interesting to see how graphs behave when using a combined loss"""
        # self.writer.add_scalar(f'{mode}Combined Loss', pred_loss + ge2e_loss, step)

    def periodic_ops(self, step):

        if step != 0:
            # save new tensorboard data to file
            if step % self.config.tb_every == 0:
                self.writer.flush()

            # Overwrite the latest version of the model
            if step % self.config.save_every == 0:
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }, self.save_path)
                
            # Make a backup
            if step % self.config.backup_every == 0:
                print("Making a backup (step %d)" % step)
                if os.path.exists(self.backup_dir):
                    os.mkdir(self.backup_dir)
                backup_fpath = os.path.join(self.backup_dir, "%s_bak_%06d.pt" % (self.config.run_id, step))
                torch.save({
                    "step": step + 1,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                }, backup_fpath)

    def tester(self):
        collater(self.dataset, self.loader, utterances_per_speaker, partials_n_frames)