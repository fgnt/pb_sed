import numpy as np
import torch
from einops import rearrange
from padertorch import Model
from padertorch.contrib.je.modules.conv import CNN2d, CNN1d
from padertorch.contrib.je.modules.features import NormalizedLogMelExtractor
from padertorch.contrib.je.modules.reduce import Mean
from torch import nn
from torchvision.utils import make_grid


class CNN(Model):
    """
    >>> config = CNN.get_config({\
            'tag_conditioning': True,\
            'cnn_2d': {'out_channels':[32,32,32], 'kernel_size': 3},\
            'cnn_1d': {'out_channels':[32,10], 'kernel_size': 3},\
            'feature_extractor': {\
                'sample_rate': 16000,\
                'fft_length': 512,\
                'n_mels': 80,\
            },\
            'false_tag_probs': 10*[.1],\
        })
    >>> sed = CNN.from_config(config)
    >>> inputs = {'stft': torch.randn((4, 1, 5, 257, 2)), 'seq_len': [5,4,3,2], 'events': torch.zeros((4,10)), 'events_alignment': torch.Tensor([0,1,1,0,0])+torch.zeros((4,10,5))}
    >>> outputs = sed(inputs)
    >>> outputs[0][0].shape
    torch.Size([4, 10, 5])
    >>> review = sed.review(inputs, outputs)
    """
    def __init__(
            self, feature_extractor, cnn_2d, cnn_1d, *,
            tag_conditioning=False, false_tag_probs=None,
            n_thresholds=101
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self._cnn_2d = cnn_2d
        self._cnn_1d = cnn_1d
        self.tag_conditioning = tag_conditioning
        if not tag_conditioning:
            assert false_tag_probs is None, false_tag_probs
        if false_tag_probs is None:
            self.register_parameter('false_tag_probs', None)
        else:
            self.register_buffer('false_tag_probs', torch.Tensor(false_tag_probs).float())
        self.n_thresholds = n_thresholds

    def cnn_2d(self, x, seq_len=None):
        if self._cnn_2d is not None:
            x, seq_len = self._cnn_2d(x, seq_len)
        if x.dim() != 3:
            assert x.dim() == 4, x.shape
            x = rearrange(x, 'b c f t -> b (c f) t')
        return x, seq_len

    def cnn_1d(self, x, seq_len=None):
        if self._cnn_1d is not None:
            x, seq_len = self._cnn_1d(x, seq_len)
        return x, seq_len

    def predict(self, x, tags=None, seq_len=None):
        x, seq_len = self.feature_extractor(x, seq_len=seq_len)
        if self.tag_conditioning:
            if self.training and self.false_tag_probs is not None:
                tag_noise = (torch.rand(tags.shape).to(x.device) < self.false_tag_probs).float() * (1 - tags)
                tags = tags + tag_noise
            x = torch.cat((x, tags[..., None, None].expand((*tags.shape, *x.shape[-2:]))), dim=1)
        h, seq_len = self.cnn_2d(x, seq_len)
        if self.tag_conditioning:
            h = torch.cat((h, tags[..., None].expand((*tags.shape, h.shape[-1]))), dim=1)
        y, seq_len = self.cnn_1d(h, seq_len)
        return (nn.Sigmoid()(y), seq_len), x

    def forward(self, inputs):
        x = inputs['stft']
        seq_len = np.array(inputs['seq_len'])
        tags = inputs['events'] if self.tag_conditioning else None
        return self.predict(x, tags, seq_len)

    def review(self, inputs, outputs):
        # compute loss
        (y, seq_len), x = outputs
        targets = inputs['events_alignment']

        assert targets.shape == y.shape, (targets.shape, y.shape)
        bce = nn.BCELoss(reduction='none')(y, targets).sum(1)
        bce = Mean(axis=-1)(bce, seq_len)
        candidate_thresholds = torch.linspace(0., 1., self.n_thresholds)
        decision = (y > candidate_thresholds[:, None, None, None].to(x.device)).float()
        tp = (decision * targets).sum((1, 3)).cpu().data.numpy()
        fp = (decision * (1.-targets)).sum((1, 3)).cpu().data.numpy()
        fn = ((1.-decision) * targets).sum((1, 3)).cpu().data.numpy()

        review = dict(
            loss=bce.mean(),
            scalars=dict(),
            histograms=dict(),
            images=dict(
                features=x[:3],
                targets=targets[:3],
            ),
            buffers=dict(
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )
        return review

    def modify_summary(self, summary):
        if 'tp' in summary['buffers']:
            k = self._cnn_1d.out_channels[-1]
            tp = np.array(summary['buffers'].pop('tp')).sum(0)
            fp = np.array(summary['buffers'].pop('fp')).sum(0)
            fn = np.array(summary['buffers'].pop('fn')).sum(0)
            p = tp / np.maximum(tp+fp, 1)
            r = tp / np.maximum(tp+fn, 1)
            f = 2*(p*r) / np.maximum(p+r, 1e-6)
            best_idx = np.argmax(f, axis=0)
            best_f = f[best_idx, np.arange(k)]
            candidate_thresholds = np.linspace(0., 1., self.n_thresholds)
            for i, idx in enumerate(best_idx):
                summary['scalars'][f'fscores/{i}'] = best_f[i]
                summary['scalars'][f'thresholds/{i}'] = candidate_thresholds[idx]
            summary['scalars']['mean_fscore'] = np.mean(best_f)

        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)

        for key, image in summary['images'].items():
            if image.dim() == 4 and image.shape[1] > 1:
                image = image[:, 0]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['feature_extractor'] = {'factory': NormalizedLogMelExtractor}
        config['cnn_2d'] = {'factory': CNN2d}
        config['cnn_1d'] = {'factory': CNN1d}
        input_size = config['feature_extractor']['n_mels']

        num_events = config['cnn_1d']['out_channels'][-1]
        if config['tag_conditioning']:
            config['cnn_2d']['in_channels'] = 1 + num_events
        else:
            config['cnn_2d']['in_channels'] = 1
        in_channels = config['cnn_2d']['in_channels']
        cnn_2d = config['cnn_2d']['factory'].from_config(config['cnn_2d'])
        output_size = cnn_2d.get_shapes((1, in_channels, input_size, 1000))[-1][2]
        input_size = cnn_2d.out_channels[-1] * output_size

        if config['tag_conditioning']:
            config['cnn_1d']['in_channels'] = input_size + num_events
        else:
            config['cnn_1d']['in_channels'] = input_size
