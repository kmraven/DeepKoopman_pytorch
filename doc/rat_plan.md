# 8×8表面電極ラット聴覚野データに対する Deep Koopman 解析計画

## 1. 解析目的

本解析では、8×8格子状の表面電極で記録されたラット聴覚野の電気生理データから、音刺激に関連する低次元潜在状態と潜在ダイナミクスを抽出する。

対象データは以下である。

* ラット数: 15個体
* 電極数: 64 ch, 8×8 grid
* サンプリング周波数: 1 kHz
* 各ブロック長: 10分
* 音条件:

  * 通常音楽
  * ガンマ音楽
  * ガンマクリック
* 区間:

  * 提示前
  * 提示中1
  * 提示中2
  * 提示後
* 総ブロック数:

[
4 \times 15 \times 3 = 180
]

本解析では、全条件・全個体に共通する3次元潜在空間を仮定し、音条件は潜在空間そのものではなく、潜在ダイナミクスに作用する入力として扱う。

つまり、

[
z_t = \phi(x_t)
]

で共通潜在状態を得て、

[
z_{t+1} = K(z_t, c_t) z_t
]

として、音条件 (c_t) が Koopman 作用素または固有値に影響するモデルを構築する。

解析で検証する主仮説は以下である。

1. 音条件によって、聴覚野活動が潜在空間内の異なる領域を占有する。
2. 音条件によって、同じ潜在状態に対するダイナミクス、すなわち固有値の性質が変化する。
3. ガンマ音楽・ガンマクリックは、通常音楽とは異なる潜在状態遷移または安定性を誘導する。

---

## 2. 基本方針

Raw 1 kHz の電位波形をそのままモデルに入力しない。

理由は、raw 波形にはラインノイズ、瞬時位相、アーチファクト、局所的な高周波変動が強く含まれ、Deep Koopman モデルが「脳状態」ではなく「電位波形の短期予測器」になりやすいためである。

したがって、モデル入力は以下のような短時間窓ごとの空間・周波数特徴とする。

[
x_t \in \mathbb{R}^{8 \times 8 \times 6}
]

ここで、8×8は電極格子、6は周波数バンド数である。

潜在空間は解釈性を優先し、3次元に固定する。

[
z_t \in \mathbb{R}^3
]

---

## 3. 前処理

### 3.1 Bad channel 検出

各10分ブロックについて、以下を検出する。

* flat channel
* 飽和チャンネル
* 極端な高分散チャンネル
* 接触不良チャンネル
* 長時間にわたり異常振幅を示すチャンネル

Bad channel は除外し、後段で空間補間する。

1ブロック内で bad channel が多すぎる場合、具体的には 64 ch 中 10 ch 以上が不良の場合、そのブロックは学習対象から除外する。

### 3.2 再参照

Bad channel を除いた上で common average reference を行う。

[
\tilde{v}_{i}(t)
================

## v_i(t)

\frac{1}{N_{\mathrm{good}}}
\sum_{j \in \mathrm{good}} v_j(t)
]

その後、bad channel は8近傍の空間補間で埋める。

### 3.3 フィルタリング

10分ブロック単位で連続信号としてフィルタリングする。

* Band-pass: 1–200 Hz
* Notch: 記録環境に応じて 50 Hz または 60 Hz とその高調波

注意点:

* 40 Hz は解析対象なので notch しない。
* 区間を細かく切ってからフィルタしない。
* フィルタは10分ブロック全体に対して適用し、エッジ効果を最小化する。

### 3.4 短時間窓化

2秒窓、0.5秒ステップで特徴量を計算する。

* Window length: 2.0 s
* Step size: 0.5 s

10分ブロックあたりの状態点数は概算で以下である。

[
\frac{600 - 2}{0.5}+1 = 1197
]

全180ブロックでは、概算で以下の状態点が得られる。

[
1197 \times 180 \approx 215000
]

ただし、これらの window は統計的に独立ではないため、評価・検定は必ずラット単位で行う。

### 3.5 周波数特徴

各 window、各チャンネルについて、以下の6バンドの power を計算する。

| Band                   | Frequency range |
| ---------------------- | --------------- |
| Delta                  | 1–4 Hz          |
| Theta                  | 4–8 Hz          |
| Alpha                  | 8–13 Hz         |
| Beta                   | 13–30 Hz        |
| Low gamma / 40 Hz band | 35–45 Hz        |
| High gamma             | 65–150 Hz       |

各 power は log 変換する。

[
x_{c,b,t}
=========

\log(P_{c,b,t} + \epsilon)
]

これにより、1時点の入力は以下となる。

[
x_t \in \mathbb{R}^{8 \times 8 \times 6}
]

### 3.6 標準化

標準化は、ラット内・チャンネル内・周波数バンド内で行う。

ただし、test leakage を避けるため、標準化パラメータは training rats の training blocks から推定し、validation/test にはその統計量を適用する。

[
x'_{c,b,t}
==========

\frac{x_{c,b,t}-\mu_{c,b}^{\mathrm{train}}}
{\sigma_{c,b}^{\mathrm{train}}}
]

### 3.7 Artifact window 除外

2秒 window ごとに以下を確認し、異常 window を除外する。

* 振幅の robust z-score
* 時間微分の robust z-score
* band power の robust z-score
* 複数チャンネル同時の急峻なアーチファクト

アーチファクトで欠損した区間をまたぐ時系列 window は作らない。

---

## 4. ラベル設計

各 window に以下のラベルを付与する。

* rat_id
* sound_condition
* section
* block_id
* time_in_block
* global_window_id

### 4.1 音条件ラベル

音条件は以下の4条件とする。

| 状態            | condition label |
| ------------- | --------------- |
| 提示前           | silence         |
| 提示中1, 通常音楽    | normal_music    |
| 提示中2, 通常音楽    | normal_music    |
| 提示中1, ガンマ音楽   | gamma_music     |
| 提示中2, ガンマ音楽   | gamma_music     |
| 提示中1, ガンマクリック | gamma_click     |
| 提示中2, ガンマクリック | gamma_click     |
| 提示後           | silence         |

提示前と提示後はいずれもモデル入力上は `silence` として扱う。

ただし、解析時には `pre` と `post` を別ラベルとして保持する。これにより、提示後の持ち越し効果を評価できる。

---

## 5. 系列データの作成

モデル学習用には、各10分ブロックから連続系列を切り出す。

系列長は64ステップとする。

[
T_{\mathrm{seq}} = 64
]

時間刻みは0.5秒なので、1系列は32秒に相当する。

[
64 \times 0.5 = 32\ \mathrm{s}
]

1つの training example は以下である。

[
(x_t, x_{t+1}, \ldots, x_{t+63})
]

系列は以下をまたいではならない。

* 異なる10分ブロック
* pre / during1 / during2 / post の境界
* artifact window による欠損区間
* 異なる音条件
* 異なるラット

---

## 6. モデル構造

## 6.1 全体構造

モデルは Deep Koopman autoencoder とする。

[
x_t \xrightarrow{\phi} z_t
]

[
z_{t+1} = K(z_t, c_t) z_t
]

[
z_t \xrightarrow{D} \hat{x}_t
]

ここで、

* (x_t): 8×8×6 の入力特徴
* (z_t): 3次元潜在状態
* (c_t): 音条件 one-hot
* (\phi): encoder
* (D): decoder
* (K): 条件依存 Koopman dynamics

### 6.2 条件入力の方針

音条件は encoder には入れない。

[
z_t = \phi(x_t)
]

音条件は固有値ネットワークにのみ入力する。

[
K_t = K(z_t, c_t)
]

この設計により、潜在空間は全条件で共通とし、音条件は流れ場・ダイナミクスに作用するものとして扱う。

---

## 7. Encoder 設計

入力は以下である。

[
8 \times 8 \times 6
]

Encoder は 2D convolutional encoder とする。

構造:

```text
Input: 8 x 8 x 6

Conv2D(32, kernel_size=3, padding="same")
GELU
LayerNorm

Conv2D(32, kernel_size=3, padding="same")
GELU
LayerNorm

Conv2D(64, kernel_size=3, padding="same")
GELU
LayerNorm

Conv2D(16, kernel_size=1, padding="same")
GELU

Flatten

Dense(64)
GELU

Dense(3)
Output: z_t in R^3
```

注意点:

* Pooling は使わない。
* 8×8 grid は小さいため、pooling による空間情報の消失を避ける。
* BatchNorm ではなく LayerNorm を使う。
* BatchNorm は条件差や個体差を batch 統計で平均化する可能性があるため避ける。

---

## 8. Decoder 設計

Decoder は encoder の mirror 構造に近いものとする。

構造:

```text
Input: z_t in R^3

Dense(64)
GELU

Dense(8 * 8 * 16)
GELU

Reshape(8, 8, 16)

Conv2D(64, kernel_size=3, padding="same")
GELU
LayerNorm

Conv2D(32, kernel_size=3, padding="same")
GELU
LayerNorm

Conv2D(6, kernel_size=3, padding="same")

Output: x_hat_t in R^{8 x 8 x 6}
```

---

## 9. 潜在ダイナミクス設計

潜在次元は3とする。

[
z_t =
[z_{1,t}, z_{2,t}, z_{3,t}]
]

3次元の内訳は以下とする。

* (z_1): 実固有値成分
* ((z_2, z_3)): 複素共役固有値ペアに対応する回転成分

潜在時間発展は以下で定義する。

[
z_{1,t+1} = \lambda_r(z_t,c_t) z_{1,t}
]

[
\begin{bmatrix}
z_{2,t+1}\
z_{3,t+1}
\end{bmatrix}
=============

\rho(z_t,c_t)
\begin{bmatrix}
\cos \theta(z_t,c_t) & -\sin \theta(z_t,c_t)\
\sin \theta(z_t,c_t) & \cos \theta(z_t,c_t)
\end{bmatrix}
\begin{bmatrix}
z_{2,t}\
z_{3,t}
\end{bmatrix}
]

ここで、固有値ネットワークは以下を出力する。

[
[\lambda_r, \rho, \theta]
=========================

g_\psi(z_t, c_t)
]

### 9.1 固有値ネットワーク

入力:

[
[z_t, c_t]
]

* (z_t): 3次元
* (c_t): 4次元 one-hot
* 入力合計: 7次元

構造:

```text
Input: concat(z_t, condition_onehot)

Dense(64)
GELU
LayerNorm

Dense(64)
GELU
LayerNorm

Dense(3)

Outputs:
lambda_r_raw
rho_raw
theta_raw
```

出力変換:

```text
lambda_r = 1.0 + 0.2 * tanh(lambda_r_raw)
rho      = exp(0.1 * tanh(rho_raw))
theta    = pi * tanh(theta_raw)
```

これにより、初期学習時に固有値が極端に発散しないよう制限する。

---

## 10. Multi-step prediction

特徴量の時間刻みは以下である。

[
\Delta t = 0.5\ \mathrm{s}
]

Prediction loss に含める最大ステップ数は以下とする。

[
S_p = 20
]

これは10秒先までの予測に対応する。

[
20 \times 0.5 = 10\ \mathrm{s}
]

Multi-step prediction は以下のように逐次的に計算する。

```text
z_pred[0] = encoder(x[t])

for h in 1..S_p:
    params_h = eigenvalue_network(z_pred[h-1], condition[t+h-1])
    K_h = build_K(params_h)
    z_pred[h] = K_h @ z_pred[h-1]
    x_pred[h] = decoder(z_pred[h])
```

Teacher forcing は使わず、予測された潜在状態を次ステップの入力に使う。

---

## 11. 損失関数

損失関数は以下とする。

[
\mathcal{L}
===========

\alpha_{\mathrm{rec}}\mathcal{L}*{\mathrm{rec}}
+
\alpha*{\mathrm{pred}}\mathcal{L}*{\mathrm{pred}}
+
\alpha*{\mathrm{lin}}\mathcal{L}*{\mathrm{lin}}
+
\alpha*{\mathrm{cov}}\mathcal{L}*{\mathrm{cov}}
+
\alpha*{\mathrm{wd}}|W|_2^2
]

### 11.1 Reconstruction loss

[
\mathcal{L}_{\mathrm{rec}}
==========================

|x_t - D(\phi(x_t))|^2
]

### 11.2 Prediction loss

[
\mathcal{L}_{\mathrm{pred}}
===========================

\frac{1}{S_p}
\sum_{h=1}^{S_p}
\left|
x_{t+h}
-------

D(z_{t+h}^{\mathrm{pred}})
\right|^2
]

### 11.3 Latent linearity loss

[
\mathcal{L}_{\mathrm{lin}}
==========================

\frac{1}{S_p}
\sum_{h=1}^{S_p}
\left|
\phi(x_{t+h})
-------------

z_{t+h}^{\mathrm{pred}}
\right|^2
]

### 11.4 Latent covariance regularization

潜在変数の collapse を避けるため、batch 内の潜在表現の共分散を正則化する。

[
\mathcal{L}_{\mathrm{cov}}
==========================

|\mathrm{Cov}(z)-I|_F^2
]

### 11.5 Loss weights

以下の値で固定する。

```text
alpha_rec  = 1.0
alpha_pred = 1.0
alpha_lin  = 0.5
alpha_cov  = 1e-3
alpha_wd   = 1e-5
```

---

## 12. 学習設定

### 12.1 データ分割

データ分割は必ずラット単位で行う。

Window 単位で train/test split してはいけない。

理由は、隣接 window が強く自己相関しており、window split ではほぼ同じ時系列断片が train と test に混入するためである。

15個体に対して、5-fold cross-validation を行う。

各 fold:

```text
test rats: 3
validation rats: 2
training rats: 10
```

### 12.2 Batch sampling

Batch size は128とする。

```text
batch_size = 128
```

各 batch には音条件が均等に含まれるようにする。

```text
silence
normal_music
gamma_music
gamma_click
```

の4条件を condition-balanced sampling する。

Silence は pre/post により数が多くなりやすいため、そのままランダムサンプリングしない。

### 12.3 Optimizer

Optimizer は AdamW とする。

```text
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-5
gradient_clip_norm = 1.0
```

### 12.4 学習段階

学習は2段階に分ける。

#### Stage 1: Autoencoder pretraining

まず autoencoder のみを学習する。

```text
epochs = 30
loss = reconstruction_loss + 1e-3 * covariance_loss + 1e-5 * weight_decay
```

#### Stage 2: Koopman training

次に Koopman loss を含めて学習する。

```text
max_epochs = 200
early_stopping_patience = 25
monitor = validation_prediction_loss
```

Stage 2 の loss は以下とする。

```text
loss =
    1.0 * reconstruction_loss
  + 1.0 * prediction_loss
  + 0.5 * latent_linearity_loss
  + 1e-3 * covariance_loss
  + 1e-5 * weight_decay
```

---

## 13. 評価指標

### 13.1 Reconstruction performance

Test rats に対して、以下の再構成誤差を評価する。

[
x_t \rightarrow z_t \rightarrow \hat{x}_t
]

評価は以下の単位で行う。

* rat
* condition
* section
* frequency band
* electrode

特に 35–45 Hz band の再構成誤差を確認する。

### 13.2 Prediction performance

Test rats に対して、以下の horizon の予測誤差を評価する。

| Horizon | Steps |
| ------- | ----- |
| 1 s     | 2     |
| 2 s     | 4     |
| 5 s     | 10    |
| 10 s    | 20    |

条件別に prediction error を比較する。

### 13.3 Latent linearity

以下が成り立つか評価する。

[
\phi(x_{t+h}) \approx z_{t+h}^{\mathrm{pred}}
]

これは単なる autoencoder ではなく、Koopman 的な潜在空間が学習されているかを確認するための指標である。

---

## 14. 解釈解析

## 14.1 潜在空間占有の比較

各 window を3次元潜在空間に写像する。

[
z_t = \phi(x_t)
]

条件ごとに以下を計算する。

* centroid
* covariance
* trajectory speed
* pre から during への移動距離
* during から post への回復距離

例:

[
\Delta z_{\mathrm{during-pre}}
==============================

## \bar{z}_{\mathrm{during}}

\bar{z}_{\mathrm{pre}}
]

ラットごとに以下を計算し、条件間で比較する。

```text
normal_music
gamma_music
gamma_click
```

### 14.2 固有値解析

固有値ネットワークの出力を解析する。

[
[\lambda_r, \rho, \theta] = g_\psi(z_t, c_t)
]

複素共役成分について、離散時間パラメータから連続時間近似を計算する。

[
\mu(z,c)=\frac{\log \rho(z,c)}{\Delta t}
]

[
f(z,c)=\frac{\theta(z,c)}{2\pi\Delta t}
]

ここでの (f) は 40 Hz neural oscillation ではない。

入力特徴が0.5秒刻みの band-power 状態であるため、ここでの (f) は数秒スケールの状態遷移周波数である。

### 14.3 Counterfactual evaluation

同じ潜在状態 (z) に対して、条件だけを入れ替えた固有値を評価する。

[
g_\psi(z,\mathrm{silence})
]

[
g_\psi(z,\mathrm{normal_music})
]

[
g_\psi(z,\mathrm{gamma_music})
]

[
g_\psi(z,\mathrm{gamma_click})
]

これにより、以下を分離する。

1. 音条件によって潜在状態 (z) の分布が変わっただけなのか
2. 同じ潜在状態 (z) に対しても、音条件がダイナミクス (K) を変えるのか

---

## 15. 統計解析

統計単位は window ではなくラットとする。

Window 数は多いが、生物学的な独立単位は15個体である。

各ラット、各音条件、各区間について以下の指標を平均する。

* latent centroid
* latent displacement
* latent trajectory speed
* prediction error
* reconstruction error
* (\lambda_r)
* (\rho)
* (\theta)
* (\mu)
* (f)

主な統計モデルは以下とする。

[
\mathrm{metric}
\sim
\mathrm{condition}
+
\mathrm{section}
+
(1|\mathrm{rat})
]

比較対象:

* gamma_music vs normal_music
* gamma_click vs normal_music
* gamma_click vs gamma_music
* during vs pre
* post vs pre

必要に応じて、ラット単位 permutation test も行う。

---

## 16. 実装上の注意

### 16.1 Leakage 防止

以下を厳守する。

* train/validation/test はラット単位で分ける。
* 標準化統計量は training rats のみから推定する。
* 隣接 window を異なる split に分けない。
* 同一ラットの別ブロックを test と train に分けない。
* artifact 除外後の欠損をまたぐ sequence を作らない。

### 16.2 潜在次元3の妥当性確認

潜在次元は解釈性のため3に固定する。

ただし、以下が不十分な場合は固有値解析を解釈しない。

* test reconstruction error が高い
* test prediction error が高い
* latent linearity が成立しない
* 35–45 Hz band の再構成が悪い

3次元モデルが十分にデータを表現できていることを確認してから、潜在状態・固有値の条件差を解釈する。

### 16.3 40 Hz の解釈

モデルの固有値から得られる周波数は、40 Hz neural oscillation そのものではない。

40 Hz 応答は入力特徴の 35–45 Hz log power に含まれる。

Koopman 固有値の周波数は、0.5秒刻みの状態系列における、より遅い状態遷移の周期性を表す。

---

## 17. 最終出力

実装では、以下を保存する。

### 17.1 前処理済みデータ

```text
processed/
  rat_id/
    block_id/
      features.npy          # shape: [time, 8, 8, 6]
      labels.csv
      artifact_mask.npy
```

### 17.2 学習済みモデル

```text
models/
  fold_0/
    encoder.pt
    decoder.pt
    eigenvalue_network.pt
    config.yaml
    metrics.json
```

### 17.3 潜在状態

```text
latents/
  fold_0/
    test_latents.parquet
```

保存する列:

```text
rat_id
block_id
section
condition
time_in_block
z1
z2
z3
lambda_r
rho
theta
mu
frequency
reconstruction_error
prediction_error_1s
prediction_error_2s
prediction_error_5s
prediction_error_10s
```

### 17.4 図

最低限、以下を出力する。

```text
figures/
  latent_3d_by_condition.png
  latent_3d_by_section.png
  latent_displacement_by_condition.png
  eigenvalue_mu_by_condition.png
  eigenvalue_frequency_by_condition.png
  prediction_error_by_horizon.png
  reconstruction_error_by_band.png
```

---

## 18. 期待される解釈

この解析で得られる解釈は以下である。

1. Encoder の出力 (z_t) により、聴覚野活動が3次元潜在空間上のどこにあるかを評価する。
2. 音条件ごとの潜在分布差により、音刺激が脳状態を異なる領域へ移動させるかを評価する。
3. 固有値ネットワーク (g_\psi(z,c)) により、音条件が潜在状態の流れ場を変えるかを評価する。
4. Counterfactual evaluation により、潜在状態分布の変化と、条件依存ダイナミクスの変化を分離する。
5. Gamma music / gamma click が normal music と異なる潜在状態遷移または安定性を示すかを検証する。
