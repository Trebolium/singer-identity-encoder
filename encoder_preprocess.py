from encoder.preprocess import preprocess_vocalset, preprocess_librispeech, preprocess_voxceleb1, preprocess_voxceleb2
from utils.argutils import print_args
from pathlib import Path
import argparse, pdb

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms and "
                    "writes them to the disk. This will allow you to train the encoder. The "
                    "datasets required are at least one of VoxCeleb1, VoxCeleb2 and LibriSpeech. "
                    "Ideally, you should have all three. You should extract them as they are "
                    "after having downloaded them and put them in a same directory, e.g.:\n"
                    "-[datasets_root]\n"
                    "  -LibriSpeech\n"
                    "    -train-other-500\n"
                    "  -VoxCeleb1\n"
                    "    -wav\n"
                    "    -vox1_meta.csv\n"
                    "  -VoxCeleb2\n"
                    "    -dev",
        formatter_class=MyFormatter
    )

    # /import/c4dm-datasets/VocalSet1-2/
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS and VoxCeleb datasets.")
    # /homes/bdoc3/my_singer_encoder/enc_preproc_outputs/vocalset
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms. If left out, "
        "defaults to <datasets_root>/SV2TTS/encoder/")
    # vocalset
    parser.add_argument("-d", "--datasets", type=str, 
                        default="librispeech_other,voxceleb1,voxceleb2", help=\
        "Comma-separated list of the name of the datasets you want to preprocess. Only the train "
        "set of these datasets will be used. Possible names: librispeech_other, voxceleb1, "
        "voxceleb2.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")
    parser.add_argument("--no_trim", action="store_true", help=\
        "Preprocess audio without trimming silences (not recommended).")
    # parser.add_argument("-ss", "--subset", type=str, default="train")
    # parser.add_argument("-sss", "--subsubset", type=str, default="clean")
    config = parser.parse_args()

    # Verify webrtcvad is available
    if not config.no_trim:
        try:
            import webrtcvad
        except:
            raise ModuleNotFoundError("Package 'webrtcvad' not found. This package enables "
                "noise removal and is recommended. Please install and try again. If installation fails, "
                "use --no_trim to disable this error message.")
    del config.no_trim

    # Process the arguments
    config.datasets = config.datasets.split(",")
    if not hasattr(config, "out_dir"):
        config.out_dir = config.datasets_root.joinpath("SV2TTS", "encoder")
    assert config.datasets_root.exists()
    config.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the datasets
    print_args(config, parser)
    preprocess_func = {
        "vocalset": preprocess_vocalset, 
        "librispeech_other": preprocess_librispeech,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
    }

    config = vars(config)
    for dataset in config.pop("datasets"):
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](**config)
