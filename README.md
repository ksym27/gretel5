# Deep避難者分布予測プロジェクト

### Gretel：https://github.com/jbcdnr/gretel-path-extrapolation<br>

### データ
場所：workspace/deep/deep1/  (火災あり)
場所：workspace/deep/deep2/　(火災なし)
*   blockage.csv：リンクの通行可能・不可
*   edges.txt：エッジ
*   nodes.txt: ノード
*   observations_6sec.txt: トラジェクトリ（ノード）
*   lengths.txt: トラジェクトリの長さ
*   paths.txt : パス（リンク）

[observations_6sec|lengths|paths]_s.txt：滞在点なし、２ステップ飛ばし

### モデルデータ
場所：workspace/chkpt/deep-nll[1|2]内のptファイル

### 設定ファイル
場所：workspace/deep/deep[1|2]/deep_nll.txt

### 学習方法
main.py workspace/deep[1|2]/deep_nll.txt

### 予測方法
evaluation.py workspace/deep[1|2]/deep_nll.txt
