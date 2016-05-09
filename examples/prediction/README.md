DEEPstationの学習結果を用いて文章の自動生成を行う
=======================================

学習済モデルを利用して、コマンドラインから文章の自動生成(prediction)を行うサンプルです。

必要なファイルの入手
-----------------

* DEEPStationの学習済みModelの画面を表示する
* 調査に利用したいEpochを選択する。
* `Download Model`ボタンを押下し、学習済モデルをダウンロードする。
* `Download Vocab File`ボタンを押下し、学習に使用した`Vocab.bin`をダウンロードする。
* `Networkタブ`で、学習に使用したNetworkをコピーし、任意のファイル名で保存する。  
   この時、ファイルは`.py`で終わる必要がある。
   
predict.pyの実行
----------------

* 前項で用意した4つのファイルを任意のディレクトリに配置する。
  * 下記のようなディレクトリ構成を想定
  
  predict  
  ├── network.py            # network   
  ├── predict.py   
  ├── vocab.bin  
  └── trained_model         # 学習済みモデル  
  
* `predict.py`のあるディレクトリで下記を実行。

  $python predict.py --model trained_model --vocabulary vocab.bin --network network.py --unit 256 --primetext 日本経済 --length 1000 --seed 354
  
* コマンドの引数に関しての詳細は `$python predict.py --help`を参照されたし。

