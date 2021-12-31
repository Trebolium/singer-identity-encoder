from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_data import partials_n_frames
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from pathlib import Path
import torch, os, pdb, time, datetime
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
        self.train_dataset = SpeakerVerificationDataset(config.clean_data_root.joinpath('train'))
        self.val_dataset = SpeakerVerificationDataset(config.clean_data_root.joinpath('val'))
        self.train_loader = SpeakerVerificationDataLoader(
            self.train_dataset,
            speakers_per_batch,
            utterances_per_speaker,
            num_workers=8,
        )
        self.val_loader = SpeakerVerificationDataLoader(
            self.val_dataset,
            speakers_per_batch,
            utterances_per_speaker,
            num_workers=8,
        )
        # Configure file path for the model
        # self.save_path = os.path.join(config.models_dir, config.run_id, config.run_id + ".pt")
        # Create the model and the optimizer
        self.model = SpeakerEncoder(self.device, self.loss_device, class_num=self.train_dataset.num_voices())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate_init)
        self.train_current_step = 0
        self.val_current_step = 0
        self.this_model_dir, self.writer = self.config_model() #no args needed as only using self params (self.config)
        # self.backup_dir_path = os.path.join(self.this_model_dir, "backups")
        self.backprop_losses = {'ge2e':0., 'pred':0., 'both':0., 'acc':0.}
        self.print_iter_metrics = {'ge2e':0., 'pred':0, 'both':0., 'acc':0.} 
        self.entire_iter_metrics = {'ge2e':0., 'pred':0., 'both':0., 'acc':0.} 
        self.mode_iters = {'train':self.config.train_iters, 'val':self.config.val_iters}
        self.print_freq = 5
        self.start_time = time.time()

    def train(self):
        # self.tester()
        training_complete = False
        while training_complete == False:
            mode = 'train'
            self.model.train()
            self.batch_iterate(self.train_loader, self.train_current_step, mode)
            self.train_current_step += self.mode_iters['train']
            mode = 'val'
            self.model.eval()
            with torch.no_grad():
                self.batch_iterate(self.val_loader, self.val_current_step, mode)
            self.val_current_step += self.mode_iters['val']
            if self.train_current_step >= self.config.stop_at_step:
                training_complete = True
        print('Training complete')

    def batch_iterate(self, loader, initial_iter_step, mode):

        # Training loop infinite as loader is infinite
        print(f'---{mode.upper()}---')
        for step, speaker_batch in enumerate(loader, initial_iter_step+1): # second enum arg means 'start counting from'
            finish_iters = (step != 0 and step % (initial_iter_step + self.mode_iters[mode]) == 0)
            should_print = (step != 0 and step % self.print_freq == 0)
            x_data_npy, y_data_npy = speaker_batch.data[0], speaker_batch.data[1]
            # Forward pass
            inputs = torch.from_numpy(x_data_npy).to(self.device).float() # speakerbatch shape = speakers, timesteps, features
            y_data = torch.from_numpy(y_data_npy).to(self.device)
            sync(self.device)
            embeds, predictions = self.model(inputs)
            accuracy = self.get_accuracy(predictions, y_data)
            ge2e_loss, pred_loss = self.get_losses(embeds, predictions, y_data) 
            metrics_list = [ge2e_loss, pred_loss, (ge2e_loss + pred_loss), accuracy]
            for i, key in enumerate(self.backprop_losses.keys()): self.backprop_losses[key] = metrics_list[i]
            for i, key in enumerate(self.print_iter_metrics.keys()): self.print_iter_metrics[key] += metrics_list[i]
            for i, key in enumerate(self.entire_iter_metrics.keys()): self.entire_iter_metrics[key] += metrics_list[i]
            self.backprop_ops(mode)
            if should_print:
                self.print_monitor(step, mode)
            if mode == 'train':
                self.periodic_ops(step)
            if finish_iters:
                self.average_print_metrics(step, mode) 
                break

                    
    def average_print_metrics(self, step, mode):

        print('AVERAGE PER ITER BLOCK')
        if mode == 'val': step = self.train_current_step # if train, step remains as step which suit conditions above. If val, model is trained to end of training stage which is now self.train_current_step 
        accuracy = round(self.entire_iter_metrics['acc']/self.mode_iters[mode], 4)
        ge2e_loss = round(self.entire_iter_metrics['ge2e'].item()/self.mode_iters[mode], 4)
        pred_loss = round(self.entire_iter_metrics['pred'].item()/self.mode_iters[mode], 4)
        both_loss = round(self.entire_iter_metrics['both'].item()/self.mode_iters[mode], 4)
        if mode == 'train': print(f'Steps {step}/{self.config.stop_at_step}, Accuracy: {accuracy}, GE2E loss: {ge2e_loss}, Pred loss: {pred_loss}, Total loss: {round(both_loss, 4)}')
        else: print(f'Steps {step}/{self.config.stop_at_step}, Accuracy: {accuracy}, GE2E loss: {ge2e_loss}, Pred loss: {pred_loss}, Total loss: {round(both_loss, 4)}')
        print()
        self.writer.add_scalar(f'Accuracy/{mode}', accuracy, step)
        self.writer.add_scalar(f'GE2E Loss/{mode}', ge2e_loss, step)
        self.writer.add_scalar(f'Class Loss/{mode}', pred_loss, step)
        self.writer.add_scalar(f'Combined Loss/{mode}', both_loss, step)
        for key in self.entire_iter_metrics.keys(): self.entire_iter_metrics[key]=0 #reset the entire_iter_metrics


    def config_model(self):

        run_id_path = os.path.join(self.config.models_dir, self.config.run_id)
        # start without creating a dir in saved models, direct tensorboard save to dummy directory
        if self.config.run_id == 'testRuns':
            print("Default model. Saving progress to testruns directory")
            # delete backups and save_paths
            writer = SummaryWriter('testRuns/test')
            return run_id_path, writer 

        else:
            # if run_id has a name...
            if os.path.exists(run_id_path): #if the run_id model exist
                if self.config.new_run_id == None: # and new_run_dir is defined
                    raise Exception('Since your run_id exists, you must also define the new save location for in the new_run_id parameter')
                else:
                    print("Model \"%s\" found, loading params." % self.config.run_id)
                    checkpoint = torch.load(os.path.join(run_id_path, 'saved_model.pt'))
                    self.train_current_step = checkpoint["step"]
                    self.model.load_state_dict(checkpoint["model_state"])
                    self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                    self.optimizer.param_groups[0]["lr"] = learning_rate_init
                    writer = SummaryWriter(comment = '_' +self.config.new_run_id)
                    new_save_dir = os.path.join(run_id_path, self.config.new_run_id)
                    os.mkdir(new_save_dir)
                    open(new_save_dir +'/config.txt', 'w').write(self.config.string_sum)
                    return new_save_dir, writer

            else: # if run_id doesn't exist
                print("No previous model found. Starting training from scratch.")
                writer = SummaryWriter(comment = '_' +self.config.run_id)
                os.mkdir(run_id_path)
                open(run_id_path +'/config.txt', 'w').write(self.config.string_sum)
                return run_id_path, writer
                


    def get_losses(self, embeds, predictions, y_data):
        # reshape output to reflect: speaker, uttrs, embeddings
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

    def backprop_ops(self, mode):
        # Backward pass
        if mode == 'train':
            self.backprop_losses[self.config.use_loss].backward() #if loss was generated without using parameters adjusted in do_gradient_ops(), there will be no gradient after backwards upon which to adjust
            self.model.do_gradient_ops(self.backprop_losses[self.config.use_loss])
            self.optimizer.step()
            self.model.zero_grad()

    def periodic_ops(self, step):

        if step != 0:
            # save new tensorboard data to file
            if step % self.config.tb_every == 0:
                self.writer.flush()

            # Overwrite the latest version of the model
            if step % self.config.save_every == 0 or step >= self.config.stop_at_step: 
                torch.save(
                    {
                    "step": step,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    },
                    os.path.join(self.this_model_dir, 'saved_model.pt'))
                print(f"Saving the model (step {step}) at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    def print_monitor(self, step, mode):
        ge2e_loss = self.print_iter_metrics['ge2e'].item()/self.print_freq
        pred_loss = self.print_iter_metrics['pred'].item()/self.print_freq
        total_loss = self.print_iter_metrics['both'].item()/self.print_freq
        pred_acc = self.print_iter_metrics['acc']/self.print_freq
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        if mode == 'train':
            print(f'Elapsed: [{et}], Steps {step}/{self.config.stop_at_step}, Accuracy: {round(pred_acc, 4)}, GE2E loss: {round(ge2e_loss, 4)}, Pred loss: {round(pred_loss, 4)}, Total loss: {round(total_loss, 4)}')
        else:
            print(f'Steps {self.train_current_step}/{self.config.stop_at_step}, Accuracy: {round(pred_acc, 4)}, GE2E loss: {round(ge2e_loss, 4)}, Pred loss: {round(pred_loss, 4)}, Total loss: {round(total_loss, 4)}') 
        for key in self.print_iter_metrics.keys(): self.print_iter_metrics[key]=0  

    def tester(self):
        collater(self.train_dataset, self.train_loader, utterances_per_speaker, partials_n_frames)    