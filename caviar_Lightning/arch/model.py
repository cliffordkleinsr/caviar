from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, optim, as_tensor
import torch
import lightning as L
from ..utilities.utils import is_listy
from .backbone import Embedding, LinBnDrop, SigmoidRange
from .depreceated import bn_drop_lin
from collections.abc import MutableSequence
from torchmetrics.classification import MulticlassAccuracy

class LitCNNTabularModel(L.LightningModule):
    "Seamless model for `Image` and `Tabular data`"
    def __init__(
            self ,
            cnn_model,
            tabular_model,
            models_outp_layers: list[int],
            ps: float = 0.0,
            out_sz: int = 0
    ) -> L.LightningModule:
         
        super().__init__()
        # self.save_hyperparameters()

        self.cnn_model = cnn_model
        self.tabular_model = tabular_model
        self.curator = MulticlassAccuracy(num_classes=2)
        ps = [0]*len(models_outp_layers) if ps is None else ps
        if not is_listy(ps): ps = [ps]*len(models_outp_layers)
        sizes = models_outp_layers + [out_sz]
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]

        new_sequential_layers = []
        for inp, outp, drop_out, act in zip(sizes[:-1], sizes[1:], ps, actns):
            individual_layers = bn_drop_lin(inp, outp, bn=True, p=drop_out, actn=act)
            new_sequential_layers += individual_layers
        self.layers = nn.Sequential(*new_sequential_layers)
        
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
            criterion = nn.CrossEntropyLoss()
            loss = criterion(z_hat, y)
            accuracy = self.curator(z_hat, y)
            metrics = {'train_loss': loss, 'train_acc': accuracy}
            self.log_dict(metrics, prog_bar=True)
            return loss
    def validation_step(self, val_batch, batch_idx) -> STEP_OUTPUT:
            x, y = val_batch
            z_hat = self(x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(z_hat, y)
            accuracy = self.curator(z_hat, y)
            metrics = {'val_loss': loss, 'valid_acc': accuracy}
            self.log_dict(metrics, prog_bar=True)
            return loss

class TabularModel(nn.Module):
    "Basic model for `Tabular data`."
    def __init__(
            self,
            emb_szs:list, # Sequence of (num_embeddings, embedding_dim) for each categorical variable
            n_cont:int, # Number of continuous variables
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
            *args,
            **kwargs
    ) -> nn.Module:
        super().__init__(*args, **kwargs)

        ps = [0]*len(layers) if ps is None else ps
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        
        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)