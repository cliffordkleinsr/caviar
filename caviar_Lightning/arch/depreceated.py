from torch import nn, optim
import torch
from .backbone import trunc_normal_
from enum import Enum
from collections.abc import MutableSequence
from torchmetrics.classification import MulticlassAccuracy
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from ..utilities.utils import is_listy
from .backbone import Embedding, LinBnDrop, SigmoidRange

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance InstanceZero')

def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0. if zero else 1.)
    return bn

def bn_drop_lin(n_in:int, n_out:int, bn:bool=True, p:float=0.0, actn=None) -> nn.Module:
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        return layers


def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm('BatchNorm', nf, ndim, zero=norm_type==NormType.BatchZero, **kwargs)

def embedding(ni:int,nf:int) -> nn.Module:
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad():
        trunc_normal_(emb.weight, std=0.01)
    return emb
    

class LitCNNTabularModel(L.LightningModule):
    def __init__(
        self ,
        cnn_model,
        tabular_model,
        out_sz:int, # Number of outputs for final `LinBnDrop` layer
        layers:list, # Sequence of ints used to specify the input and output size of each `LinBnDrop` layer
        ps:float|MutableSequence=None, # Sequence of dropout probabilities for `LinBnDrop`
        embed_p:float=0.0, # Dropout probability for `Embedding` layer
        y_range=None, # Low and high for `SigmoidRange` activation 
        use_bn:bool=True, # Use `BatchNorm1d` in `LinBnDrop` layers
        bn_final:bool=False, # Use `BatchNorm1d` on final layer
        bn_cont:bool=True, # Use `BatchNorm1d` on continuous variables
        act_cls=nn.ReLU(inplace=True), # Activation type for `LinBnDrop` layers
        lin_first:bool=True, # Linear layer is first or last in `LinBnDrop` layers 
    ) -> L.LightningModule:
        
        super().__init__()
        self.cnn_model = cnn_model
        self.tabular_model = tabular_model
        self.curator = MulticlassAccuracy(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()
        ps = [0]*len(layers) if ps is None else ps
        if not is_listy(ps): ps = [ps]*len(layers)
        sizes = layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]

        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first)
                    for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x) -> None:
        x_image = self.cnn_model(x[0])# image
        x_tab = self.tabular_model(*x[1])# tabular, unpack categorical and continous data
        # concatenate the outputs
        x = torch.cat([x_image, x_tab], 1)
        # pass through fully connected layers
        x = self.layers(x)
        return x
    def configure_optimizers(self) -> OptimizerLRScheduler:
            optimizer = optim.Adam(self.parameters(), lr=1e-4)
            return optimizer
    def training_step(self, train_batch, batch_idx) -> STEP_OUTPUT:
            x, y = train_batch
            z_hat = self(x)
            loss =  self.criterion(z_hat, y)
            accuracy = self.curator(z_hat, y)
            metrics = {'train_loss': loss, 'train_acc': accuracy}
            self.log_dict(metrics, prog_bar=True)
            return loss
    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
            x, y = val_batch
            z_hat = self(x)
            loss = self.criterion(z_hat, y)
            accuracy = self.curator(z_hat, y)
            metrics = {'val_loss': loss, 'valid_acc': accuracy}
            self.log_dict(metrics, prog_bar=True , on_epoch=True)
            return loss
    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
            return self(batch)