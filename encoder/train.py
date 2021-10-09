from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_data import partials_n_frames
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch, os, pdb
from torch import nn
from encoder.tester import collater
from torch.utils.tensorboard import SummaryWriter

def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool, stop_at_step: int, notes: str, this_model_dir: Path, string_sum: str):


    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=8,
    )
    
    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")
    
    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device, dataset.num_speakers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = os.path.join(this_model_dir, run_id + ".pt")
    backup_dir = os.path.join(this_model_dir, "backups")

    # Load any existing model
    if run_id == 'testrun':
        # delete backups and state_fpaths
        writer = SummaryWriter('testRuns/test')
        print("Default model. Saving progress to testruns directory")
    else:
        if not force_restart:
            if os.path.exists(state_fpath):
                checkpoint = torch.load(state_fpath)
                init_step = checkpoint["step"]
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                optimizer.param_groups[0]["lr"] = learning_rate_init
                writer = SummaryWriter(comment = '_' +run_id)
                print("Model \"%s\" found, resuming training." % run_id)
            else:
                writer = SummaryWriter(comment = '_' +run_id)
                if not os.path.exists(this_model_dir):
                    os.mkdir(this_model_dir)
                open(this_model_dir +'/config.txt', 'a').write(string_sum)
                print("Starting training from scratch.")
        else:
            print('Force restart enabled!')
            writer = SummaryWriter(comment = '_' +run_id)
            if not os.path.exists(this_model_dir):
                os.mkdir(this_model_dir)
            open(this_model_dir +'/config.txt', 'a').write(string_sum)
            print("Starting training from scratch.")

    model.train()



    # Initialize the visualization environment
    # vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    # vis.log_dataset(dataset)
    # vis.log_params()
    # device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    # vis.log_implementation({"Device": device_name})
    
    # Training loop
    profiler = Profiler(summarize_every=10, disabled=False)
    # collater(dataset, loader, utterances_per_speaker, partials_n_frames)


    training_complete = False
    while training_complete == False:
        for step, speaker_batch in enumerate(loader, init_step):
            # step += 1 #this method will not translate well if continuing from a previous training session
            print(step)
            profiler.tick("Blocking, waiting for batch (threaded)")
            
            x_data_npy, y_data_npy = speaker_batch.data[0], speaker_batch.data[1]
            # Forward pass
            inputs = torch.from_numpy(x_data_npy).to(device).float() # speakerbatch shape = speakers, timesteps, features
            y_data = torch.from_numpy(y_data_npy).to(device)
            sync(device)
            profiler.tick("Data to %s" % device)
            embeds, predictions = model(inputs)
            sync(device)
            profiler.tick("Forward pass")
            # reshape output to reflect: speaker, uttrs, embeddings
            embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
            # this is the loss for one pass, reflecting 64 speakers of 10 utterances
            loss, eer = model.loss(embeds_loss)
            pred_loss = nn.functional.cross_entropy(predictions, y_data)
            final_loss = pred_loss + loss
            sync(loss_device)
            profiler.tick("Loss")

            # Backward pass
            model.zero_grad()
            final_loss.backward() #if loss was generated without using parameters adjusted in do_gradient_ops(), there will be no gradient after backwards upon which to adjust
            profiler.tick("Backward pass")
            model.do_gradient_ops(final_loss)
            optimizer.step()
            profiler.tick("Parameter update")

            writer.add_scalar("Ge2e Loss", loss, step)
            writer.add_scalar('Classification Loss', pred_loss, step)
            writer.add_scalar('Total Loss', final_loss, step)

            # Update visualizations
            learning_rate = optimizer.param_groups[0]["lr"]
            # vis.update(loss.item(), eer, step)
            
            # Draw projections and save them to the backup folder
            # if umap_every != 0 and step % umap_every == 0:
            #     print("Drawing and saving projections (step %d)" % step)
            #     backup_dir.mkdir(exist_ok=True)
            #     projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
                # embeds = embeds.detach().cpu().numpy()
                # vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
                # vis.save()

            if vis_every != 0 and step % vis_every == 0:
                writer.flush()

            # Overwrite the latest version of the model
            if save_every != 0 and step % save_every == 0:
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)
                
            # Make a backup
            if backup_every != 0 and step % backup_every == 0:
                print("Making a backup (step %d)" % step)

                os.mkdir(backup_dir)
                backup_fpath = os.path.join(backup_dir, "%s_bak_%06d.pt" % (run_id, step))
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, backup_fpath)
            
            if stop_at_step !=0 and step == stop_at_step:
                print('Training complete. Stopping session...')
                training_complete = True
                break