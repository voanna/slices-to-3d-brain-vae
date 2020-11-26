__author__ = 'voanna'

import os
import stat

learning_rates = [1e-3]
latent_dimensions = [512]

reconstruction_data_loss_weights = [1.0]
kl_latent_loss_weights = [0.2]

standardize_latent_curve_fit_losses = [True]
batch_sizes = [128]

seen_dict = {}
EXPERIMENT = 'MICCAI-release-version'

gen_dir = os.path.join('experiments', EXPERIMENT, 'gen')
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)

data_dir = os.path.join('experiments', EXPERIMENT, 'gen', 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

tensorboard_dir = os.path.join('experiments', EXPERIMENT, 'gen', 'tensorboard')
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

log_dir = os.path.join('experiments', EXPERIMENT, 'gen', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_dir = '/home/voanna/curvature/' + log_dir

scripts_dir = os.path.join('experiments', EXPERIMENT, 'gen', 'scripts')
if not os.path.exists(scripts_dir):
    os.makedirs(scripts_dir)

train_all_sh = os.path.join(scripts_dir, 'train_all_64.sh')



with open(train_all_sh, 'w') as g:
    g.write('#!/bin/sh\n')
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for kl_latent_loss_weight in kl_latent_loss_weights:
                run_name = 'lr_{}' \
                           '_kl_{}_' \
                           '_bsize_{}' \
                           ''.format(
                    learning_rate,
                    kl_latent_loss_weight,
                    batch_size
                )

                try:
                    if seen_dict[run_name]:
                        continue
                except KeyError:
                    seen_dict[run_name] = True

                print(run_name)

                pycmd = 'python experiments/{}/main_experiment_64.py ' \
                        ' --learning-rate={}' \
                        ' --kl-latent-loss-weight={} ' \
                        ' --batch-size={} ' \
                        ''.format(
                    EXPERIMENT,
                    learning_rate,
                    kl_latent_loss_weight,
                    batch_size,
                )
                shname = os.path.join(scripts_dir, run_name + '.sh')
                logfile = os.path.join(log_dir, run_name + '.log')

                with open(shname, 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('source .bashrc\n')
                    f.write('echo $(hostname)\n')
                    f.write('cd curvature \n')
                    f.write('echo $(pwd) \n')
                    f.write('echo $(date) \n')
                    f.write('source set-cuda-pytorch.sh\n')
                    f.write(pycmd + '\n')
                    f.write('echo $(date) \n')
                st = os.stat(shname)
                os.chmod(shname, st.st_mode | stat.S_IEXEC)

                g.write(
                    'qsub -l gpu -l h_vmem={}G -l h_rt=12:00:00 -j y -o '.format(40) + logfile + ' ' + shname + '\n')

st = os.stat(train_all_sh)
os.chmod(train_all_sh, st.st_mode | stat.S_IEXEC)
