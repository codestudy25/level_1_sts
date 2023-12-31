import argparse
import random
from utils.utils import read_json 
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from itertools import product
import os
import glob

import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchmetrics    
import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from pytorch_lightning.loggers import WandbLogger

from utils.utils import *
from models.model import *
# from models.callbacks import *
from data.data_module import *
#from models.loss import *
os.environ["TZ"] = "Asia/Seoul"

def main(config:Dict):
    
    #seed 고정
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    random.seed(config["seed"])
    
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', default=config["inference"], action="store_true")
    parser.add_argument('--best', default=config["best"], action="store_true") # https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true
    parser.add_argument('--test', default=config["test"], action="store_true")
    parser.add_argument('--ensemble', default=config["ensemble"], action="store_true")
    parser.add_argument('--resume', default=config["resume"], action="store_true")
    parser.add_argument('--shuffle', default=config["shuffle"], action="store_true")
    parser.add_argument('--wandb_project_name', default=config["wandb_project_name"], type=str)
    parser.add_argument('--wandb_username', default=config["wandb_username"], type=str)
    parser.add_argument('--model_name', default=config["model_name"], type=str)
    parser.add_argument('--model_detail', default=config["model_detail"], type=str) # 돌리는 모델의 detail 추가
    parser.add_argument('--batch_size', default=config["batch_size"], type=int)
    parser.add_argument('--max_epoch', default=config["max_epoch"], type=int)
    parser.add_argument('--learning_rate', default=config["learning_rate"], type=float)
    parser.add_argument('--kfold', default=config["kfold"], type=int)
    parser.add_argument('--data_dir', default=config["data_dir"])
    parser.add_argument('--model_dir', default=config["model_dir"])
    parser.add_argument('--test_output_dir', default=config["test_output_dir"])
    parser.add_argument('--output_dir', default=config["output_dir"])
    parser.add_argument('--train_path', default=config["train_path"])
    parser.add_argument('--dev_path', default=config["dev_path"])
    parser.add_argument('--test_path', default=config["test_path"])
    parser.add_argument('--predict_path', default=config["predict_path"])

    args = parser.parse_args()
    
    train_path = Path(args.data_dir) / args.train_path
    dev_path = Path(args.data_dir) / args.dev_path
    test_path = Path(args.data_dir) / args.test_path
    predict_path = Path(args.data_dir) / args.predict_path
    
    
    def train():
        print("Start training...")
        
        model_name = args.model_name
        model_detail = args.model_detail
        
        grids = list(product(args.batch_size, args.learning_rate))
        print(f"Total {len(grids)} combinations has been detected...")
        
        for i, combination in enumerate(grids, start=1):
            batch_size, learning_rate = combination
            max_epoch = args.max_epoch
            
            print(f"#{i}" + "="*80)
            print(f"model_name: {model_name}, model_detail: {model_detail}\nbatch_size: {batch_size}\nmax_epoch: {max_epoch}\nlearning_rate: {learning_rate}\n")
            
            latest_version, _ = get_latest_version(args.model_dir, model_name, model_detail)
            
            # model_dir / model_provider / model_name + model_version + batch_size + max_epoch + learning_rate + current_epoch + current_step + eval_metric + YYYYMMDD + HHMMSS + .ckpt
            # ./saves/klue/roberta-small_v03_16_1_1e-05_000_00583_0.862_20231214_221830.ckpt 같은 형태로 저장됩니다
            save_name = "-".join(model_name.split("/")[1].split()) + "_"+ "-".join(model_detail.split()) + prefix_zero(latest_version+1, 2) + f"_{batch_size}_{max_epoch}_{learning_rate}"
            print(f"save_name: {save_name}")
            
            wandb_logger = WandbLogger(project=args.wandb_project_name, entity=args.wandb_username)

            early_stop_custom_callback = EarlyStopping("val_pearson", patience=5, verbose=True, mode="max")
            
            #model_provider = model_name.split("/")[0] # "klue"/roberta-large
            #dirpath = Path(args.model_dir) / model_provider
            dirpath = Path(args.model_dir)
            dirpath.mkdir(parents=True, exist_ok=True)

            checkpoint_callback = ModelCheckpoint(
                    monitor="val_pearson",
                    save_top_k=1,
                    dirpath=dirpath,
                    filename=save_name + "_{epoch:03d}_{step:05d}_" + "{val_pearson:0.3f}" + "_" + datetime.today().strftime("%Y%m%d_%H%M%S"),
                    auto_insert_metric_name=False,
                    save_weights_only=False,
                    verbose=True,
                    mode="max"
                )

            loss_fns = [nn.SmoothL1Loss()]
            
            model = Model(model_name, learning_rate, loss_fns)
            
            num_folds = args.kfold
            
            split_seed = config["seed"]
            if num_folds > 1:
                print(f"KFold dataloader will be used. nums_folds: {num_folds}, split_seed: {split_seed}")
                results = []
                
                for k in range(num_folds):
                    print(f"Current fold: {k}th fold" + "=" * 80)
                    kfdataloader = KFoldDataloader(model_name, batch_size, args.shuffle, train_path, dev_path, test_path, predict_path,
                                                        k=k, split_seed=split_seed, num_splits=num_folds)
                    kfdataloader.prepare_data()
                    kfdataloader.setup()
                    
                    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch//num_folds, 
                                             callbacks=[checkpoint_callback,early_stop_custom_callback],
                                             log_every_n_steps=1,logger=wandb_logger)

                    trainer.fit(model=model, datamodule=kfdataloader)
                    score = trainer.test(model=model, datamodule=kfdataloader)
                    
                    results.extend(score)
                
                result = [x['test_pearson'] for x in results]
                score = sum(result) / num_folds
                print(f"K fold Test score: {score}" + "=" * 80)
                
            else:
                dataloader = Dataloader(model_name, batch_size, args.shuffle, train_path, dev_path, test_path, predict_path)
                
                #special token의 embedding을 학습에 포함시킵니다?
                
                trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=max_epoch, 
                                         callbacks=[checkpoint_callback,early_stop_custom_callback],
                                         log_every_n_steps=1,logger=wandb_logger)
                
                # Train part
                trainer.fit(model=model, datamodule=dataloader)
                trainer.test(model=model, datamodule=dataloader)
                
            
            # 학습이 완료된 모델을 저장합니다.
            torch.save(model, dirpath / f"{save_name}.pt")
    
    def inference():
        print("Start inference...")
        
        model_name = args.model_name
        model_detail = args.model_detail
        
        if args.best:
            print("Loading the best performance model...")
            select_version, select_version_perf, select_version_path = get_version(args.model_dir, model_name, model_detail, best=True)
        else:
            print("Loading latest trained model...")
            select_version, select_version_perf, select_version_path = get_version(args.model_dir, model_name, model_detail)
        batch_size = int(select_version_path.stem.split("_")[-8])
        
        print(f"#inference" + "=" * 80)
        print(f"model_name: {model_name}\nversion: v{select_version}\nval_perf: {select_version_perf}\nbatch_size: {batch_size}\n")

        trainer = pl.Trainer(accelerator="gpu", 
                             devices=1, max_epochs=1)
        model = Model.load_from_checkpoint(select_version_path)
        
        output_dir = Path(args.output_dir) if not args.test else Path(args.test_output_dir)
        #model_provider = model_name.split("/")[0] # "klue"/roberta-large
        #output_path = output_dir / model_provider
        output_path = output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        
        if args.test:
            print(f"\nInference on test dataset {test_path}...")
            dataloader = Dataloader(model_name, batch_size, False, train_path, dev_path, test_path, test_path) # prediction with dev.csv
            test_predictions = trainer.predict(model=model, datamodule=dataloader)
            test_predictions = list(round(val.item(), 1) for val in torch.cat(test_predictions))
            
            # Aggregate batch outputs into one
            output = pd.read_csv(test_path)
            output["predict"] = test_predictions
            output = output.drop(columns=["binary-label"])
            metric = torchmetrics.functional.pearson_corrcoef(torch.tensor(output["predict"]), torch.tensor(output["label"]))
            output_file_name = '_'.join(select_version_path.stem.split("_")[:-3]) + f"_{metric:.3f}_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"  
        else:
            print(f"\nInference for submission {predict_path}...")
            dataloader = Dataloader(model_name, batch_size, False, train_path, dev_path, test_path, predict_path)
            predictions = trainer.predict(model=model, datamodule=dataloader)
            predictions = list(round(val.item(), 1) for val in torch.cat(predictions)) # (# batches, batch_size * 1) -> (# batches * batch_size * 1)
            
            output = pd.read_csv("/data/ephemeral/home/sample_submission.csv")
            output["target"] = predictions
            output_file_name = '_'.join(select_version_path.stem.split("_")[:-2]) + f"_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv" # add prediction time
        output.to_csv(output_path / output_file_name, index=False)

    def ensemble():
        print("Create ensemble result...")
        ensemble_dir = Path("/data/ephemeral/home/github_codestudy25/ensembles")
        if not ensemble_dir.exists():
            raise ValueError("Ensemble directory does not exist.")
        model_paths = list(ensemble_dir.rglob('*.ckpt'))
        
        if len(model_paths) < 2:
            raise ValueError("At least two models are required for ensemble.")
        print(f"Total {len(model_paths)} models are detected...")
        
        output_dir = Path(args.test_output_dir) if args.test else Path(args.output_dir)
        output_path = output_dir / "ensemble"
        output_path.mkdir(parents=True, exist_ok=True)
        
        ensemble_names = []
        model_predictions = []
        for i, model_path in enumerate(model_paths):
            model_name = "/".join([model_path.parent.name, model_path.stem.split("_")[0]])
            model_metric = float(model_path.stem.split("_")[-3])
            batch_size = int(model_path.stem.split("_")[-8])
            ensemble_names.append("_".join([str(i), model_name.split("/")[1], str(model_metric), str(batch_size)]))
            print(f"Processing {i}th model: {model_name}...")

            if args.test:
                print(f"\nEnsemble on test dataset {test_path}...")
                dataloader = Dataloader(model_name, batch_size, False, train_path, dev_path, test_path, test_path)
            else:
                print(f"\nEnsemble for submission {predict_path}...")
                dataloader = Dataloader(model_name, batch_size, False, train_path, dev_path, test_path, predict_path)
            
            trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1)
            model = Model.load_from_checkpoint(model_path)

            predictions = trainer.predict(model=model, datamodule=dataloader)
            # predictions = list(val.item() for val in torch.cat(predictions))
            predictions = torch.cat(predictions).squeeze() # take off batch dimension
            model_predictions.append(predictions)
            
        # voting using softmax
        model_predictions = torch.stack(model_predictions, dim=0)
        model_scores = torch.nn.functional.softmax(model_predictions, dim=0)
        # print(f"Model predictions: {model_predictions.shape}, Model scores: {model_scores.shape}")
        assert model_predictions.shape == model_scores.shape
        # adopt score as weith
        model_results = model_predictions * model_scores # element-wise (weighted sum)
        model_results = model_results.sum(dim=0)
        # dealing with out-of-range values
        model_results = torch.where(model_results<0, 0, model_results)
        model_results = torch.where(model_results>5, 5, model_results)
        
        if args.test:
            ensemble_names.append(f"{len(model_paths)}_Ensemble_0.000_00")
            #  Plot results
            plot_models(ensemble_names, torch.cat((model_predictions, model_results.unsqueeze(0)), dim=0), test_path, "label", error_gap=1.5)
            # Aggregate batch outputs into one
            output = pd.read_csv(test_path)
            output["predict"] = model_results
            output = output.drop(columns=["binary-label"])
            metric = torchmetrics.functional.pearson_corrcoef(torch.tensor(output["predict"]), torch.tensor(output["label"]))
            #output_file_name = "_".join(ensemble_names) + f"_{metric:.3f}_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            output = pd.read_csv("/data/ephemeral/home/sample_submission.csv")
            output["target"] = predictions
            #output_file_name = "_".join(ensemble_names) + f"_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"
        #output.to_csv(output_path / output_file_name, index=False)
        output.to_csv(output_path/config['output_name'],index=False)

    if args.inference:
        if args.ensemble:
            ensemble()
        else:
            inference()
    else:
        train()
        
    
if __name__ == '__main__':
    config = read_json('/data/ephemeral/home/github_codestudy25/level_1_sts/config/config.json')
    main(config=config)
        
            