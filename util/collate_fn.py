import math
import random
import torch
import numpy as np


class CollateFn(object):
    def __init__(self, frame_num, sample_type='fixed_ordered'):
        sample_type = sample_type
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = frame_num

        self.frames_all_limit = -1

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        seqs_batch, labs_batch, info_batch = [], [], [[], []]

        for bt in batch:
            seqs_batch.append(bt[0])
            labs_batch.append(bt[1])
            info_batch[0].append(bt[2][0])
            info_batch[1].append(bt[2][1])

        global count
        count = 0

        def sample_frames(seqs):
            global count
            sampled_fras = []
            seq_len = len(seqs)
            indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    raise ValueError('Number of input frames must be fixed!')

                if self.ordered:
                    fs_n = frames_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    assert len(idx_lst) == frames_num
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num

                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

                for i in indices:
                    sampled_fras.append(seqs[i])
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]

        if self.sampler == "fixed":
            fras_batch = [np.asarray(fras_batch[i]) for i in range(batch_size)]  # [f, b]
        else:
            raise ValueError('Number of input frames must be fixed!')

        batch = [np.stack(fras_batch), labs_batch, info_batch]
        batch[0] = torch.from_numpy(batch[0])
        batch[1] = torch.tensor(batch[1])
        batch[2][0] = torch.tensor(batch[2][0])
        return batch
