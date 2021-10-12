import os, shutil, sys, pdb

"""This script assumes that new vocalset features are preplaced in the way that the encoder_preprocess.py file puts them in
It looks like they are placed in folders named to describing all conditions except vowels.
We look for strings with the spkr_ids in the directory title and send these to a fresh directory with JUST the singer title, deleting the original afterwards
Ww also concatenate the source files together, which tell us the original audio path and the corresponding feature file"""

spkr_ids = ['_male1_','_male2','_male3','_male4','_male5','_male6','_male7','_male8','_male9','_male10','_male11','female1','female2','female3','female4','female5','female6','female7','female8','female9']
r, d, _ = next(os.walk(sys.argv[1]))
for subdir in d:
    if '.' in subdir:
        shutil.rmtree(os.path.join(r, subdir)) 
        continue
    if subdir not in spkr_ids:
        for spkr_id in spkr_ids:
            spkr_id_path = os.path.join(sys.argv[1], spkr_id)
            if not os.path.exists(spkr_id_path):
                os.mkdir(spkr_id_path)
            if spkr_id in subdir:
                _,_,files = next(os.walk(os.path.join(r, subdir)))
                for f in files:
                    f_path = os.path.join(r, subdir, f)
                    if os.path.exists(f_path) and f_path.endswith(".txt"):
                        d_t_file = open(os.path.join(spkr_id_path, f), 'a')
                        s_t_file = open(f_path, 'r')
                        d_t_file.write(s_t_file.read())
                        d_t_file.close()
                        s_t_file.close()
                        os.remove(f_path)
                        print(f'copied contents from {f_path} to {os.path.join(spkr_id_path, f)}')
                    else:
                        try:
                            shutil.move(f_path, spkr_id_path)
                        except:
                            print(f'couldn\'t move {f_path} to {spkr_id_path}!')
                            pdb.set_trace()
        try:
            os.rmdir(os.path.join(r, subdir))
            print(f'removed {os.path.join(r, subdir)} from root')
        except:
            pdb.set_trace()