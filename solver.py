from data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from model import SpeakerEncoder
from pathlib import Path
import torch, os, time, datetime, sys, yaml, math, pdb
from torch import nn
from tester import collater
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(1, '/homes/bdoc3/my_utils')
from neural.scheduler import EarlyStopping

def sync(device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

"""
SIE:
SingerIdentityEncoder is an object that initiates a model based on user inputs
The SIE's train method is implemented which in turn calls its own methods hierarchy:
    train
        tester (optional for dev testing)
        batch_iterate
            average_print_metrics
            get_losses
            get_accuracy
            backprop_ops
            print_monitor
"""
class SingerIdentityEncoder:

    def __init__(self, config, feat_params=None) -> None:
        #Initialise configurations
        self.print_freq = 5
        self.train_current_step = 0
        self.val_current_step = 0
        self.config = config
        self.device = torch.device(f'cuda:{self.config.which_cuda}' if torch.cuda.is_available() else "cpu")
        self.loss_device = torch.device("cpu")
        self.prev_lowest_ge2e = math.inf

        #Load feature parameters from dataset yaml
        if feat_params == None:
            with open(os.path.join(self.config.feature_dir, 'feat_params.yaml')) as File:
                feat_params = yaml.load(File, Loader=yaml.FullLoader)
        self.feat_params = feat_params
        self.num_feats = feat_params['num_feats']
        #Create dataset and dataloader for val, train subsets
        self.train_dataset = SpeakerVerificationDataset(config.feature_dir.joinpath('train'),
            config, feat_params
        )
        self.val_dataset = SpeakerVerificationDataset(config.feature_dir.joinpath('val'),
            config, feat_params
        )
        self.train_loader = SpeakerVerificationDataLoader(
            self.train_dataset,
            config.speakers_per_batch,
            config.utterances_per_speaker,
            config.num_timesteps,
            self.num_feats,
            num_workers=32,
        )
        self.val_loader = SpeakerVerificationDataLoader(
            self.val_dataset,
            config.speakers_per_batch,
            config.utterances_per_speaker,
            config.num_timesteps,
            self.num_feats,
            num_workers=32,
        )

        # Create the model and the optimizer
        self.model = SpeakerEncoder(self.device, self.loss_device,
            self.train_dataset.num_voices(),
            self.num_feats,
            config.model_hidden_size,
            config.model_embedding_size,
            config.num_layers
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate_init)
        self.this_model_dir, self.writer = self.config_model()
        self.backprop_losses = {'ge2e':0., 'pred':0., 'both':0., 'acc':0.}
        self.print_iter_metrics = {'ge2e':0., 'pred':0, 'both':0., 'acc':0.} 
        self.entire_iter_metrics = {'ge2e':0., 'pred':0., 'both':0., 'acc':0.} 
        self.mode_iters = {'train':self.config.train_iters, 'val':self.config.val_iters}
        self.EarlyStopping = EarlyStopping(patience=config.patience)
        self.start_time = time.time()

    #Cycle between train and eval mode until completion, monitoring training steps
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

    # iterate through a specificed subset, performing forward/backward passes and other implementing other methods
    def batch_iterate(self, loader, initial_iter_step, mode):

        #Infinite training loop (as loader is infinite) until break
        print(f'---{mode.upper()}---')
        for step, speaker_batch in enumerate(loader, initial_iter_step+1):
            # print(time.time() - self.start_time)
            finish_iters = (step != 0 and step % (initial_iter_step + self.mode_iters[mode]) == 0)
            should_print = (step != 0 and step % self.print_freq == 0)
            x_data_npy, y_data_npy = speaker_batch.data[0], speaker_batch.data[1]

            # Forward pass
            inputs = torch.from_numpy(x_data_npy).to(self.device).float() # speakerbatch shape = speakers, timesteps, features
            y_data = torch.from_numpy(y_data_npy).to(self.device)
            sync(self.device)
            embeds, predictions = self.model(inputs)

            # generate loss metrics
            accuracy = self.get_accuracy(predictions, y_data)
            ge2e_loss, pred_loss = self.get_losses(embeds, predictions, y_data)
            metrics_list = [ge2e_loss, pred_loss, (ge2e_loss + pred_loss), accuracy]
            for i, key in enumerate(self.backprop_losses.keys()): self.backprop_losses[key] = metrics_list[i]
            for i, key in enumerate(self.print_iter_metrics.keys()): self.print_iter_metrics[key] += metrics_list[i]
            for i, key in enumerate(self.entire_iter_metrics.keys()): self.entire_iter_metrics[key] += metrics_list[i]
            self.backprop_ops(mode)

            # scheduled functions: print progress and metrics, save model
            if should_print:
                self.print_monitor(step, mode)
            self.periodic_ops(step, ge2e_loss)
            if finish_iters:
                self.average_print_metrics(step, mode) 
                break

    
    # print out metrics of model performance in a human-readable way and saving to tensorbaord format
    def average_print_metrics(self, step, mode):

        print('AVERAGE PER ITER BLOCK')
        if mode == 'val': step = self.train_current_step # if train, step remains as step which suit conditions above. If val, model is trained to end of training stage which is now self.train_current_step 
        accuracy = round(self.entire_iter_metrics['acc']/self.mode_iters[mode], 4)
        ge2e_loss = round(self.entire_iter_metrics['ge2e'].item()/self.mode_iters[mode], 4)
        pred_loss = round(self.entire_iter_metrics['pred'].item()/self.mode_iters[mode], 4)
        both_loss = round(self.entire_iter_metrics['both'].item()/self.mode_iters[mode], 4)
        if mode == 'train':
            print(f'Steps {step}/{self.config.stop_at_step}, Accuracy: {accuracy}, GE2E loss: {ge2e_loss}, Pred loss: {pred_loss}, Total loss: {round(both_loss, 4)}')
        else:
            print(f'Steps {step}/{self.config.stop_at_step}, Accuracy: {accuracy}, GE2E loss: {ge2e_loss}, Pred loss: {pred_loss}, Total loss: {round(both_loss, 4)}')
            if self.EarlyStopping.check(ge2e_loss):
                print(f'Early stopping employed.')
                self.periodic_ops(self.config.stop_at_step+1, ge2e_loss)
                exit(0)
        print()
        self.writer.add_scalar(f'Accuracy/{mode}', accuracy, step)
        self.writer.add_scalar(f'GE2E Loss/{mode}', ge2e_loss, step)
        self.writer.add_scalar(f'Class Loss/{mode}', pred_loss, step)
        self.writer.add_scalar(f'Combined Loss/{mode}', both_loss, step)
        for key in self.entire_iter_metrics.keys(): self.entire_iter_metrics[key]=0 #reset the entire_iter_metrics


    # Use user inputs to initiate mode, decide whether to save, or include pretrained weights
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
                    self.optimizer.param_groups[0]["lr"] = self.config.learning_rate_init
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
                

    # generate loss metrics from classifications and embeddings
    def get_losses(self, embeds, predictions, y_data):
        embeds_loss = embeds.view((self.config.speakers_per_batch, self.config.utterances_per_speaker, -1)).to(self.loss_device) # reshape output to reflect: speaker, uttrs, embeddings
        # this is the loss for one pass, reflecting 64 speakers of 10 utterances
        ge2e_loss, _ = self.model.loss(embeds_loss)
        pred_loss = nn.functional.cross_entropy(predictions, y_data)
        return ge2e_loss, pred_loss

  
    # generate classification loss using predictions and labels
    def get_accuracy(self, predictions, y_data):
        _, predicted = torch.max(predictions.data, 1)
        correct_preds = (predicted == y_data).sum().item()
        sync(self.loss_device)
        return correct_preds / len(y_data) #accuracy

 
    # backpropogate through the network and reset gradients
    def backprop_ops(self, mode):
        if mode == 'train':
            self.backprop_losses[self.config.use_loss].backward() #if loss was generated without using parameters adjusted in do_gradient_ops(), there will be no gradient after backwards upon which to adjust
            self.model.do_gradient_ops(self.backprop_losses[self.config.use_loss])
            self.optimizer.step()
            self.model.zero_grad()

 
    # check number of steps in training cycle to determine if saving model and flush TB
    def periodic_ops(self, step, ge2e_loss):

        if step != 0:
            # save new tensorboard data to file
            if step % self.config.tb_every == 0:
                self.writer.flush()
            # Overwrite the latest version of the model
            if ge2e_loss < self.prev_lowest_ge2e or step >= self.config.stop_at_step: 
                torch.save(
                    {
                    "step": step,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    },
                    os.path.join(self.this_model_dir, 'saved_model.pt'))
                print(f"Saving the model (step {step}) at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
                self.prev_lowest_ge2e = ge2e_loss

    # print metric info in human-readable format
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


    # method for testing the computation of feature and batches without multiprocessing for debugging
    def tester(self):
        collater(self.train_dataset, self.config.utterances_per_speaker, self.config.num_timesteps, self.num_feats, self.config, self.feat_params)    