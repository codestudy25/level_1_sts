import argparse
import random

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from pytorch_lightning.loggers import WandbLogger

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"]}
        else:
            return {"input_ids": self.inputs[idx]["input_ids"],
                    "attention_mask": self.inputs[idx]["attention_mask"]}, torch.tensor(self.targets[idx]) 

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)

        # source에 해당하는 special token
        special_tokens_dict = {
            "additional_special_tokens": [
                "[petition]",
                "[nsmc]",
                "[slack]",
                "[sampled]",
                "[rtt]",
            ]
        }

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tok=len(self.tokenizer) # 늘어난 vocab 개수를 나중에 반영해주기 위해


        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.source_columns = ['source']


    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            s1, s2 = item[self.source_columns].item().split('-')

            text=f'[{s1}]'+f'[{s2}]'+text # source token을 추가

            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)

            for key in outputs:
                outputs[key] = torch.tensor(outputs[key], dtype=torch.long)

            data.append(outputs)
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist() # 이 부분이 별거 아닌것 같지만 겁나 중요함
            # 여기서 data[]의 []내부에 ['label'] 이라는 리스틀르 넣어주는데, 이럴 경우에 targets는 [[0],[1],[0]...] 이런 식으로  추가된다.
            # 즉, shape 측면에서 ( n,1 )이 되는 것이고 -> 이 것이 Dataset에 들어가게 되면
            # 최종적으로 Dataset의 __getitem__에서 target에 해당하는 출력의 shape이 (batch_size, 1)이 된다.
            # 이렇게 되면 loss function을 계산할 때, logits의 shape은 [batch_size, 1] target의 shape은 [batch_size,1]이 되고
            # loss의 결과 reduction 'none'을 적용하면 [batch_size, 1]의 shape을 가지게 된다.
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,shuffle=args.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


def custom_loss(logits, targets,loss_func,penalty):
    # logits : [batch_size, 1]
    # targets : [batch_size, 1]
    original_loss = loss_func(reduction='none')(logits, targets) # [batch_size, 1]

    loss_mask = (logits > 5) # [batch_size, 1]
    customed_loss = original_loss * loss_mask * penalty # [batch_size, 1]

    loss = original_loss + customed_loss # [batch_size, 1]
    return loss 


class Model(pl.LightningModule):
    def __init__(self, model_name, lr,penalty):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.penalty = penalty

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        
        self.plm.resize_token_embeddings(dataloader.tok) # vocab size를 늘려줍니다

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss

    def forward(self, **x):
        x = self.plm(**x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch # [batch_size,1]
        


        logits = self(**x) # [batch_size, 1]

        loss = torch.mean(custom_loss(logits, y.float(), self.loss_func, self.penalty))

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)
        loss = self.loss_func()(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(**x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(**x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--model_detail', default='v', type=str) # 돌리는 모델의 detail 추가

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--penalty', default=0.01, type=float)

    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--dev_path', default='./data/dev.csv')
    parser.add_argument('--test_path', default='./data/dev.csv')
    parser.add_argument('--predict_path', default='./data/test.csv')
    args = parser.parse_args()

    # sweep config 설정
    sweep_config={
        'method':'random',
        'parameters':{'penalty':{'distribution': 'uniform', 'min': 0.001, 'max': 0.2}}, # 'lr':{'distribution':'uniform', 'min':1e-6, 'max':1e-4},'batch_size':{'values':[16,32,64]},
        'metric':{'name':'val_pearson', 'goal':'maximize'},
        
    }

    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate, args.penalty)


    # sweep 함수
    def sweep_train(config=None):
        early_stop_custom_callback = EarlyStopping(
        "val_pearson", patience=3, verbose=True, mode="max"
        )

        checkpoint_callback = ModelCheckpoint(
        monitor="val_pearson",
        save_top_k=1,
        dirpath="./",
        filename='_'.join(args.model_name.split() + args.model_detail.split()), # model에 따라 변화
        save_weights_only=False,
        verbose=True,
        mode="max",
        )

        wandb.init(config=config)
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
        model = Model(args.model_name, args.learning_rate,args.penalty)
        wandb_logger = WandbLogger(project='baseline', entity='gypsi12',name='_'.join(args.model_name.split() + args.model_detail.split()
        )) # model에 따라 변화

        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, callbacks=[checkpoint_callback,early_stop_custom_callback],log_every_n_steps=1,logger=wandb_logger)
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

    # sweep 생성
    sweep_id=wandb.sweep(sweep_config,project='baseline',entity='gypsi12')
    wandb.agent(sweep_id=sweep_id,function=sweep_train,count=10)
