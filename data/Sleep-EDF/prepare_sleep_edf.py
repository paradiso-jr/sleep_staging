# @author: sungju2121@naver.com
# modified by jr
import logging
import argparse
import pyedflib
import numpy as np

from pathlib import Path
# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
MOVE = 5
UNK = 6

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK
}

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-i",
                        type=str, 
                        default="./edf",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", "-o",
                        type=str, 
                        default="./npz",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", 
                        nargs='+',
                        default= ['EEG Fpz-Cz'],
                        choices=['EEG Fpz-Cz', 'EEG Pz-Oz','EOG horizontal', 'Resp oro-nasal','EMG submental','Temp rectal',],
                        help="Name of the channels in the dataset.")
    parser.add_argument("--log_file", 
                        type=str, 
                        default="info_ch_extract.log",
                        help="Log file.")
    parser.add_argument("--w_edge_mins",
                        type=int,
                        default=30,
                        help="Time duration before and after fall asleep")
    args = parser.parse_args()

    for select_ch in list(args.select_ch):
        in_dir = Path(args.data_dir)
        out_dir = Path(args.output_dir) / \
                Path("w_edge_{}_mins".format(args.w_edge_mins)) /\
                Path(f"{select_ch.split(' ')[0]}")
        
        # if target channel have been preprocessced, then continue
        print(out_dir)
        if out_dir.exists():
            continue
        
        out_dir.mkdir(parents=True)
        
        log_file = out_dir / Path(args.log_file)

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        
        # Read raw and annotation from EDF files
        psg_fnames = list(in_dir.glob("*PSG.edf"))
        ann_fnames = list(in_dir.glob("*Hypnogram.edf"))
        psg_fnames.sort()
        ann_fnames.sort()
        psg_fnames = np.asarray(psg_fnames)
        ann_fnames = np.asarray(ann_fnames)
        
        for i in range(len(psg_fnames)):
            logger.info("Loading ...")
            logger.info("Signal file: {}".format(str(psg_fnames[i])))
            logger.info("Annotation file: {}".format(str(ann_fnames[i])))

            psg_f = pyedflib.EdfReader(str(psg_fnames[i]))
            ann_f = pyedflib.EdfReader(str(ann_fnames[i]))
            
            assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
            start_datetime = psg_f.getStartdatetime()
            logger.info("Start datetime: {}".format(str(start_datetime)))

            file_duration = psg_f.getFileDuration()
            logger.info("File duration: {} sec".format(file_duration))
            
            epoch_duration = psg_f.datarecord_duration
            if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
                epoch_duration = epoch_duration / 2
                logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
            else:
                logger.info("Epoch duration: {} sec".format(epoch_duration))
                
            
            # Extract signal from the selected channel
            ch_names = psg_f.getSignalLabels()
            ch_samples = psg_f.getNSamples()
            select_ch_idx = -1
            for s in range(psg_f.signals_in_file):
                if ch_names[s] == select_ch:
                    select_ch_idx = s
                    break
            if select_ch_idx == -1:
                raise Exception("Channel not found.")
            
            sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
            n_epoch_samples = int(epoch_duration * sampling_rate)
            signals = psg_f.readSignal(select_ch_idx).reshape(-1, n_epoch_samples)
            logger.info("Select channel: {}".format(select_ch))
            logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx]))
            logger.info("Sample rate: {}".format(sampling_rate))
            
            # Sanity check
            n_epochs = psg_f.datarecords_in_file
            if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
                n_epochs = n_epochs * 2
            assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

            # Generate labels from onset and duration annotation
            labels = []
            total_duration = 0
            ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
            for a in range(len(ann_stages)):
                onset_sec = int(ann_onsets[a])
                duration_sec = int(ann_durations[a])
                ann_str = "".join(ann_stages[a])
                
                # Sanity check
                assert onset_sec == total_duration
                
                # Get label value
                label = ann2label[ann_str]
                
                # Compute # of epoch for this stage
                if duration_sec % epoch_duration != 0:
                    logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                    raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
                duration_epoch = int(duration_sec / epoch_duration)

                # Generate sleep stage labels
                label_epoch = np.ones(duration_epoch, dtype=int) * label
                labels.append(label_epoch)

                total_duration += duration_sec

                logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                    onset_sec, duration_sec, label, ann_str
                ))
            labels = np.hstack(labels)        
            # Remove annotations that are longer than the recorded signals
            labels = labels[:len(signals)]

            # Get epochs and their corresponding labels
            x = signals.astype(np.float32)
            y = labels.astype(np.int32)
            
            # Select only w_edge_mins minutes before and after sleep periods
            w_edge_mins = args.w_edge_mins
            nw_idx = np.where(y != stage_dict["W"])[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0: start_idx = 0
            if end_idx >= len(y): end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx+1)
            logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
            x = x[select_idx]
            y = y[select_idx]
            logger.info("Data after selection: {}, {}".format(x.shape, y.shape))

            # Remove movement and unknown
            move_idx = np.where(y == stage_dict["MOVE"])[0]
            unk_idx = np.where(y == stage_dict["UNK"])[0]
            if len(move_idx) > 0 or len(unk_idx) > 0:
                remove_idx = np.union1d(move_idx, unk_idx)
                logger.info("Remove irrelavant stages")
                logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
                logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
                logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
                logger.info("  Data before removal: {}, {}".format(x.shape, y.shape))
                select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
                x = x[select_idx]
                y = y[select_idx]
                logger.info("  Data after removal: {}, {}".format(x.shape, y.shape))
            
            filename = psg_fnames[i].name.replace("-PSG.edf", ".npz")
            save_dict = {"x": x, 
                         "y": y, 
                         "fs": sampling_rate,
                         "ch_label": select_ch,
                         "w_edge_mins": w_edge_mins,
                         "start_datetime": start_datetime,
                         "file_duration": file_duration,
                         "epoch_duration": epoch_duration,
                         "n_all_epochs": n_epochs,
                         "n_epochs": len(x),}
            np.savez(out_dir / filename, **save_dict)
            
            logger.info("\n=======================================\n")
if __name__ == '__main__':
    main()
