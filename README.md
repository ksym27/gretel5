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
場所：workspace/chkpt/deep-nll2内のptファイル
ダウンロードしてください<br>
https://drive.google.com/drive/folders/1bZjxtmBnyybLxXoBZLBNuu4zTWk-gGyv?usp=sharing

### 設定ファイル
場所：workspace/deep/deep2/deep_nll.txt

### 学習方法
main.py workspace/deep2/deep_nll.txt

### 予測方法
deep_pred.py workspace/deep2/deep_nll.txt<br>
main_loop関数<br>
- start_time.txt：sim開始秒
- end_time.txt:　sim終了秒
- step_time.txt：simステップ秒

output dir:workspace/chkpt/deep-nll2
- pred_observations.txt：内部用データ
- pred_observation_times.txt：内部用データ
- pred_observation_steps.txt：内部用データ
- pred_condition.txt：内部用データ
- pred_nodes.txt:IDごとの予測されたノードIDのリスト
- pred_times.txtIDごとの予測されたノードIDに対応する発災後の時間（秒）
- prediction_result.txt：(id,nodeid,time,status)形式のデータ、可視化用


