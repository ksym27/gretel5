# Deep避難者分布予測プロジェクト

### Gretel：https://github.com/jbcdnr/gretel-path-extrapolation<br>

### 修正点
*   道路の閉塞を考慮する。
*   時間を考慮する。（まだ終わってない）

### 設定ファイル
場所：workspace/deep/deep/deep_nll.txt

### データ
場所：workspace/deep/deep/
*   blockage.csv：リンクの通行可能・不可
*   edges.txt：エッジ
*   nodes.txt: ノード
*   observations_6sec.txt: トラジェクトリ（ノード）
*   lengths.txt: トラジェクトリの長さ
*   paths.txt : パス（リンク）

### 学習方法
main.py workspace/deep/deep_nll.txt

### 予測方法
evaluation.py workspace/deep/deep_nll.txt
