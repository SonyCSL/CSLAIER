DEEPstationの学習結果を用いて画像の予測を行う
=======================================

学習済モデルを利用して、コマンドラインから画像の調査(inspection)を行うサンプルです。

必要なファイルの入手
-----------------

* DEEPStationの学習済みModelの画面を表示する
* 調査に利用したいEpochを選択する。
* `Download Model`ボタンを押下し、学習済モデルをダウンロードする。
* `Download Label`ボタンを押下し、学習に使用した`labels.txt`をダウンロードする。
* `Download Mean File`ボタンを押下し、学習に使用した`mean.npy`をダウンロードする。
* `Networkタブ`で、学習に使用したNetworkをコピーし、任意のファイル名で保存する。  
   この時、ファイルは`.py`で終わる必要がある。
   
inspect.pyの実行
----------------

* 前項で用意した4つのファイルを任意のディレクトリに配置する。
  * 下記のようなディレクトリ構成を想定
  
  inspection  
  ├── network.py            # network  
  ├── image_to_inspect.jpg  # 調査したい画像  
  ├── inspect.py  
  ├── labels.txt  
  ├── mean.npy  
  └── trained_model         # 学習済みモデル  
  
* `inspect.py`のあるディレクトリで下記を実行。

  $python main.py image_to_inspect.jpg network.py trained_model
  
* コマンドの引数に関しての詳細は `$python main.py --help`を参照されたし。

