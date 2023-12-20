import pandas as pd
import torch
from pathlib import Path

# ensemble_outputs라는 폴더를 만들고 그 폴더 내부에 있는 csv파일로 앙상블 진행
# scoreboard.txt라는 파일을 만들고, 그 파일안에는 '파일명:점수' 형태로 저장

class Ensemble():
    def __init__(self):
        self.output_dir_files = list(Path('/data/ephemeral/home/github_codestudy25/level_1_sts/to_ensemble_files').iterdir())
        self.score_files = [file for file in Path('/data/ephemeral/home/github_codestudy25/level_1_sts/scoreboard.txt').read_text().split('\n') if file != '']
        self.score_dict={file.split(':')[0].strip():file.split(':')[1].strip() for file in self.score_files} # 파일명:점수

    def voting(self):
        num_files = len(self.output_dir_files)
        
        scores = torch.Tensor([float(self.score_dict[file.name[:-4]]) for file in self.output_dir_files]) # file.name[:-4] : 파일명에서 .csv 제외, softmax를 하기 위한 Tensor형태로 저장 
        scores = torch.nn.functional.softmax(scores,dim=-1)

        #outputs = [pd.read_csv(file)['target'] for file in self.output_dir_files]
        outputs = [pd.read_csv(file)['label'] for file in self.output_dir_files]
        
        ensemble_output = [outputs[i]*scores[i].item() for i in range(num_files)] # 각 파일의 target에 softmax를 취한 score를 곱해줌, ,item() : Tensor를 float로 변환

        ensemble_output = pd.concat(ensemble_output, axis=1) # 행 방향으로 concat
        ensemble_output = pd.Series(ensemble_output.sum(axis=1)) # 행방향으로 sum, 이후, submission Dataframe 의 taret에 위치시키기 위한 Series 형태로 저장

        # traget 범위를 0에서 5 사이로 제한
        for i in range(len(ensemble_output)):
            if ensemble_output.iloc[i] > 5:
                ensemble_output.iloc[i] = 5
            elif ensemble_output.iloc[i] < 0:
                ensemble_output.iloc[i] = 0

        output = pd.read_csv('/data/ephemeral/home/sample_submission.csv')
        output['target'] = ensemble_output
        output.to_csv('/data/ephemeral/home/github_codestudy25/level_1_sts/ensemble_outputs/ensemble_output.csv', index=False)

if __name__ == '__main__':
    ensemble = Ensemble()
    ensemble.voting()