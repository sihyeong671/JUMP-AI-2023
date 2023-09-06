import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
from deepchem.models.torch_models import GCNModel
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    features = ["AlogP", "Molecular_Weight", "Num_H_Acceptors", "Num_H_Donors", "Num_RotatableBonds", "LogD", "Molecular_PolarSurfaceArea"]
    mlm_target = "MLM"
    hlm_target = "HLM"

    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    train_df["AlogP"] = np.where(pd.isna(train_df["AlogP"]), train_df["LogD"], train_df["AlogP"])
    train_df['mol'] = train_df['SMILES'].apply(lambda x : Chem.MolFromSmiles(x))
    train_df['mol'] = train_df['mol'].apply(lambda x: Chem.AddHs(x))
    train_df['num_of_atoms'] = train_df['mol'].apply(lambda x: x.GetNumAtoms())
    train_df['num_of_heavy_atoms'] = train_df['mol'].apply(lambda x: x.GetNumHeavyAtoms())

    test_df["AlogP"] = np.where(pd.isna(test_df["AlogP"]), test_df["LogD"], test_df["AlogP"])
    test_df['mol'] = test_df['SMILES'].apply(lambda x : Chem.MolFromSmiles(x))
    test_df['mol'] = test_df['mol'].apply(lambda x: Chem.AddHs(x))
    test_df['num_of_atoms'] = test_df['mol'].apply(lambda x: x.GetNumAtoms())
    test_df['num_of_heavy_atoms'] = train_df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    def canonize(mol):
        return Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True)

    canon_smile = []
    for molecule in train_df['SMILES']:
        canon_smile.append(canonize(molecule))
        
    train_df['canon_smiles'] = canon_smile
    train_df.info()


    from rdkit.Chem import rdFingerprintGenerator
    rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator()
    train_df["Morgan_FPs"] = train_df["mol"].apply(lambda x: rdkgen.GetFingerprintAsNumPy(x))


    def smiles_split(df, smiles, seed=42, k_fold=5, splitter='scaffold'):
        Xs, ys = np.arange(len(smiles)), np.ones(len(smiles))
        dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=ys,w=np.zeros(len(smiles)),ids=smiles)
        if splitter == 'random':
            splitter = dc.splits.RandomSplitter()
        elif splitter == 'scaffold':
            splitter = dc.splits.ScaffoldSplitter()
        elif splitter == 'fingerprints':
            splitter = dc.splits.FingerprintSplitter()
        folds = splitter.k_fold_split(dataset, k=k_fold, seed=seed)
        dfs = []
        for fold in folds:
            train_indices = fold[0].X
            val_indices = fold[1].X
            train_df = df.iloc[train_indices].reset_index(drop=True)
            val_df = df.iloc[val_indices].reset_index(drop=True)
            dfs.append((train_df, val_df))
        return dfs

    BATCH_SIZE=32
    SEED=42
    K_FOLD=5

    train_smiles = train_df['SMILES'].tolist()


    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    for train_fold, val_fold in smiles_split(train_df, train_smiles, seed=SEED, k_fold=K_FOLD, splitter='fingerprints'):
        break

    for col in ['AlogP','Molecular_Weight','Num_H_Acceptors','Num_H_Donors','Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea']:
        scaler = MinMaxScaler()
        train_fold[col] = scaler.fit_transform(train_fold[[col]].values).reshape(-1)
        val_fold[col] = scaler.transform(val_fold[[col]].values).reshape(-1)
        test_df[col] = scaler.transform(test_df[[col]].values).reshape(-1)
        
        train_nan_value = train_fold[col].mean()
        train_fold[col] = train_fold[col].fillna(train_nan_value)
        val_fold[col] = val_fold[col].fillna(train_nan_value)
        test_df[col] = test_df[col].fillna(train_nan_value)



    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    test_df['MLM']=0
    test_df['HLM']=0



    train_X = featurizer.featurize(train_fold['SMILES'].tolist())
    train_w = train_fold[['AlogP','Molecular_Weight','Num_H_Acceptors','Num_H_Donors','Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea']].values
    train_dataset = dc.data.NumpyDataset(X=train_X, y=train_fold[['MLM','HLM']].values, w=train_w)

    val_X = featurizer.featurize(val_fold['SMILES'].tolist())
    val_w = val_fold[['AlogP','Molecular_Weight','Num_H_Acceptors','Num_H_Donors','Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea']].values
    val_dataset = dc.data.NumpyDataset(X=val_X, y=val_fold[['MLM','HLM']].values, w=val_w)

    test_X = featurizer.featurize(test_df['SMILES'].tolist())
    test_w = test_df[['AlogP','Molecular_Weight','Num_H_Acceptors','Num_H_Donors','Num_RotatableBonds', 'LogD', 'Molecular_PolarSurfaceArea']].values
    test_dataset = dc.data.NumpyDataset(X=test_X, y=test_df[['MLM','HLM']].values, w=test_w)


    # In[132]:


    def collate_fn(samples):
        X = [sample[0] for sample in samples]
        y = torch.Tensor([sample[1] for sample in samples])
        w = torch.Tensor([sample[2] for sample in samples])
        return ([X],y,w)


    # In[133]:


    from torch.utils.data import DataLoader

    train_datas = []
    val_datas = []
    test_datas = []

    for x,y,w in zip(train_dataset.X, train_dataset.y, train_dataset.w):
        train_datas.append((x,y,w))
        
    for x,y,w in zip(val_dataset.X, val_dataset.y, val_dataset.w):
        val_datas.append((x,y,w))
        
    for x,y,w in zip(test_dataset.X, test_dataset.y, test_dataset.w):
        test_datas.append((x,y,w))
        
    train_dataloader = DataLoader(train_datas, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_datas, batch_size=BATCH_SIZE*2, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_datas, batch_size=BATCH_SIZE*2, collate_fn=collate_fn)


    # In[146]:


    import lightning as L
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW

    class LitMPNNSMILESClassification(L.LightningModule):
        def __init__(self, batch_size, node_out_feats=64, n_tasks=2):
            super().__init__()
            model = GCNModel(mode='regression', n_tasks=n_tasks,
                node_out_feats=node_out_feats,
                batch_size=batch_size,
                learning_rate=0.001
            )
            self._prepare_batch = lambda batch : model._prepare_batch(batch)
            self.model = model.model
            self.model.model.predict = nn.Identity()
            self.batch_size=batch_size
            self.classifier = nn.Sequential(
                nn.LazyLinear(node_out_feats//2),
                nn.ReLU(),
                nn.LazyLinear(n_tasks)
            )
            
            self.validation_step_outputs = []
            
        def forward(self, x, xp):
            x = self.model(x)
            x = torch.cat([x, xp], dim=-1)
            x_out = self.classifier(x)
            return x_out
        
        def training_step(self, batch, batch_idx):
            x, *_ = self._prepare_batch(batch)
            y_true, xp = batch[1]/100, batch[2]
            y_pred = self(x, xp)
            loss1 = F.mse_loss(y_pred[:,0].flatten(), y_true[:,0].flatten())
            loss2 = F.mse_loss(y_pred[:,1].flatten(), y_true[:,1].flatten())
            loss = (loss1**0.5 + loss2**0.5)/2
            self.log_dict({"train_loss": loss}, on_step=True, prog_bar=True, batch_size=self.batch_size)
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, *_ = self._prepare_batch(batch)
            y_true, xp = batch[1], batch[2]
            y_pred = self(x, xp)*100
            loss1 = F.mse_loss(y_pred[:,0].flatten(), y_true[:,0].flatten())
            loss2 = F.mse_loss(y_pred[:,1].flatten(), y_true[:,1].flatten())
            loss = (loss1, loss2)
            self.validation_step_outputs.append(loss)
            return loss
        
        def on_validation_epoch_end(self):
            loss = torch.Tensor(self.validation_step_outputs)
            loss1, loss2 = loss[:, 0], loss[:, 1]
            loss = ((loss1.mean())**0.5 + (loss2.mean())**0.5)/2
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.validation_step_outputs.clear()
            
        def predict_step(self, batch, batch_idx):
            x, *_ = self._prepare_batch(batch)
            y_true, xp = batch[1], batch[2]
            y_pred = self(x, xp)*100
            return y_pred
            
        def configure_optimizers(self): 
            optimizer = AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
            return optimizer


    # In[147]:


    lit_model = LitMPNNSMILESClassification(
        batch_size=BATCH_SIZE
    )


    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoint/',
        filename='MPNN-{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}',
        save_top_k=1,
    )

    trainer = L.Trainer(
        accelerator='gpu',
        # strategy='ddp_spawn',
        max_epochs=100,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(lit_model, train_dataloader, val_dataloader)