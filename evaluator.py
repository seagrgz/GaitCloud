import torch
import subprocess
import torch.nn.functional as F

DEFAULT_ATTRIBUTES = ('memory.total','memory.free')
class MetricEvaluator():
    def __init__(self, dist_func=None):
        if dist_func is None:
            self.dist_func = self.ComputeDistance
        else:
            self.dist_func = dist_func

    def rank_1_accuracy(self, probe_embeds, probe_labels, gallery_embeds=None, gallery_labels=None, logger=None):
        """
            embeddings: [n, c, p]
            labels:     [n]
        """
        if gallery_embeds is None:
            gallery_embeds = probe_embeds
            gallery_labels = probe_labels
            g_is_p = True
        else:
            g_is_p = False
        probe_embeds = probe_embeds.permute(dims=(2,0,1)).contiguous().float()
        gallery_embeds = gallery_embeds.permute(dims=(2,0,1)).contiguous().float()
        dist_metric_part = self.dist_func(probe_embeds.to('cpu'), gallery_embeds.to('cpu'), logger) #[p, n_p, n_g]
        #dist_metric_part = self.dist_func(probe_embeds, gallery_embeds, logger) #[p, n_p, n_g]
        dist_metric = dist_metric_part.mean(0) #[n_p, n_g]
        del dist_metric_part
        if g_is_p:
            dist_metric = dist_metric.fill_diagonal_(dist_metric.max())
        rank_1_indices = torch.argmin(dist_metric, dim=1)
        del dist_metric
        predict_labels = gallery_labels[rank_1_indices] #[n_p]
        matches = probe_labels[probe_labels == predict_labels]
        failed_indics = torch.nonzero(probe_labels != predict_labels).squeeze()
        accuracy = matches.shape[0]/probe_labels.shape[0]
        return accuracy, failed_indics

    def ComputeDistance(self, x, y, logger):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        #x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        ##logger.info('x2 calculated: {}'.format(self.get_gpu_info()[0]['memory.free']))
        #y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        ##logger.info('y2 calculated: {}'.format(self.get_gpu_info()[0]['memory.free']))
        #inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        ##logger.info('inner calculated: {}'.format(self.get_gpu_info()[0]['memory.free']))
        #dist = x2 + y2 - 2 * inner
        #logger.info('dist calculated: {}'.format(self.get_gpu_info()[0]['memory.free']))
        dist = torch.sum(x ** 2, -1).unsqueeze(2) + torch.sum(y ** 2, -1).unsqueeze(1) - 2 * x.matmul(y.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist.detach()))  # [p, n_x, n_y]
        return dist

    def get_gpu_info(self, nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
        nu_opt = '' if not no_units else ',nounits'
        cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
        output = subprocess.check_output(cmd, shell=True)
        lines = output.decode().split('\n')
        lines = [ line.strip() for line in lines if line.strip() != '' ]
        return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]
