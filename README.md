# NSF-GAN
音声波形生成モデルNeural source-filterに敵対的学習を導入し，ダイナミクスの改善と音声品質の向上を行った．
b-NSFをGeneratorに，DiscriminatorにMelGANのMulti-scale Discriminatorを採用した．

### 論文
- [b-NSF](https://arxiv.org/abs/1810.11946)
- [H-sinc-NSF(従来法)](https://arxiv.org/abs/1908.10256)
- [MelGAN](https://arxiv.org/abs/1910.06711)

### 参考にしたソースコード
- [NSF](https://github.com/mitsu-h/project-NN-Pytorch-scripts)
- [MelGANのDiscriminator](https://github.com/kan-bayashi/ParallelWaveGAN)
## Requirements

# Usage
## 音声データのダウンロード
本研究ではLJSpeech Datasetを使用した．\
[dataset](https://keithito.com/LJ-Speech-Dataset/)
## サンプリングレートの変換
現状のハイパーパラメータはサンプリングレートが16kHzである必要があるため，
`downsample_lj.py`を実行して，LJSpeech Datasetの音声を22.05kHzから16kHzへと
ダウンサンプリングする．引数は[ソースコード](downsample_lj.py)参照
## 学習データの分割
`div_dataset.py`を用いて，データセットを学習・検証・テストデータへと分割する．
分割は，ファイル名をテキストファイルに保存，必要に応じてそのテキストファイルに記載の音声ファイルを読み込む．
引数は[ソースコード](div_dataset.py)参照
## 学習データの平均・標準偏差の算出
ネットワークでメルスペクトログラム・基本周波数の標準化を行うため，`calc_data_mean_std.py`で，
学習データの平均・標準偏差を計算，pklファイルへと保存する．\
pklファイルのパスを，`config.json`の`mean_std_path`に入力する．
引数は[ソースコード](calc_data_mean_std.py)参照
## 学習
`train.py`でNSF-GANの学習を行う．ハイパーパラメータは`config.json`で設定を行う．
引数は[ソースコード](train.py)参照
### tensorboardによる可視化
`config.json`記載の`log_event_path`にtensorboardのログを出力しているので，以下で学習の確認を行う．
```
tensorboard --logdir=<log_event_path>
```
### 従来法の学習
`train_h_sinc_nsf.py`で従来法であるH-sinc-NSFの学習を行う．
ハイパーパラメータは`config_h_sinc_nsf.json`で設定を行う．

###テストデータの推論
`inference.py`で，分割したテストデータのメルスペクトログラム，基本周波数を入力とした合成音をすべて出力する
引数は[ソースコード](inference.py)参照

##その他ソースコード
- calc_db.py\
音声のダイナミクスを定量評価するために，指定のフォルダ内にある全音声のRMSで算出した音圧レベルの平均を出力する．
- cp_txt_list.py\
テストデータの自然音声をデータセットから抽出する．主観評価実験などで実際の音声データが必要な場合に用いる


