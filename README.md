# 🎯 Facial Expression Recognition using StyleGAN2 - Classification Improvement Plan
**期間:** 2025/11/10 ～ 2026/01/10（約8週間）  
**目的:** pSp Encoder で得た W⁺ 潜在特徴からの表情分類モデル（CNN / Transformer）を改善し、Macro-F1 > 0.70 を目標にする。  
**比較対象:** ResEmoteNet (SoTA, augmentationなし)

---

## 🧩 A. 全体目標
1. 既存分類器（CNN, Transformer）の性能向上（正規化・損失関数・構造改善）
2. 公平比較のために **ResEmoteNet** を同一条件で再現
3. 論文用に **Macro-F1 / Accuracy / Confusion Matrix / 有意差検定** をまとめる
4. augmentationなし設定での SoTA 比較を行い、論文表に掲載可能な結果を得る

---

## 📅 B. 8週間スケジュール（タスク別）

### 🕐 Week 1 — セットアップ・再現性基盤
- [ ] AffectNet-HQ（8:2 split）の前処理統一（リサイズ・正規化・ラベルマップ）
- [ ] ResEmoteNet リポジトリをクローン・依存関係整備  
  - [ ] GitHub実装の動作確認（デモスクリプト実行）
  - [ ] 自データセットへの入力適合（同一前処理）
- [ ] 既存CNN / Transformer / SVM の再訓練とログ確認（seed固定）
- [ ] baseline（CNN=0.6364 / Transformer=0.6309 / SVM=0.65）を保存

---

### 🕑 Week 2 — SoTA (ResEmoteNet) 再現
- [ ] augmentation なしで ResEmoteNet を再学習
- [ ] 評価（Macro-F1, Accuracy, confusion matrix）
- [ ] 自作モデルとの初期比較表を作成（同条件比較）

---

### 🕒 Week 3 — 改良案①（構造・正規化）
- [ ] Transformer に Learnable Positional Encoding を追加
- [ ] CNN に BatchNorm、Transformer に LayerNorm を導入
- [ ] Label Smoothing (ε=0.1) 導入
- [ ] 各構成で 3-seed 実験を実施 → 平均と標準偏差を算出

---

### 🕓 Week 4 — 改良案②（学習制御・正則化）
- [ ] Optimizer を AdamW に変更、Scheduler を CosineAnnealing に
- [ ] 学習率スイープ（1e-3～3e-5）、weight decay, dropout の最適化
- [ ] Focal Loss (γ=2) の比較実験
- [ ] 最良設定を3つまで選定

---

### 🕔 Week 5 — 改良案③（潜在空間操作）
- [ ] W⁺ に Gaussian ノイズ (σ=1e-2) を付与して再訓練
- [ ] 潜在Mixup（λ∼Beta(0.4,0.4)）を適用
- [ ] CNN-Transformer ハイブリッド構成（局所→文脈）を実装・評価

---

### 🕕 Week 6 — 集約・最適構成の検証
- [ ] Week3〜5の有効手法を組み合わせた最終モデルを構築
- [ ] 5ラン実験を実施（平均・標準偏差）
- [ ] ResEmoteNet（同条件）との比較表を作成

---

### 🕖 Week 7 — 結果整理・可視化
- [ ] ハイパラ一覧表（lr, batch, dropout, optimizer, seed）
- [ ] 各モデルの混同行列 / Precision / Recall を算出
- [ ] 学習曲線（train-loss / val-F1）をプロット
- [ ] 結果ログ・チェックポイントを整理

---

### 🕗 Week 8 — 論文用まとめ・統計検定
- [ ] augmentationなし条件での最終比較表（Macro-F1）
- [ ] paired t-test または bootstrap による有意差検定
- [ ] augmentationあり条件の補足実験（別表に掲載）
- [ ] LaTeX テーブル / 図生成
- [ ] 最終報告書・論文セクション（Result, Discussion）草案作成

---

## 📊 C. 評価指標
- **主要:** Macro-F1  
- **補助:** Accuracy, Precision/Recall per class, Confusion Matrix  
- **安定性:** 平均 ± 標準偏差（3～5 seed）  
- **統計検定:** paired bootstrap または t-test (p < 0.05)

---

## ⚙️ D. ログ管理・再現性
- [ ] 実験名・seed・ハイパラを自動記録 (Hydra / W&B / TensorBoard)
- [ ] Checkpoint 保存ルール (`best_model.pth`, `config.yaml`)
- [ ] 全モデルを同一seed条件で再評価可能にする

---

## 📁 E. 出力予定物（論文用）
- Table: Model vs Macro-F1 (augmentationなし)
- Figure: Confusion Matrix / F1 curve
- Appendix: ハイパラ設定表、学習曲線、統計検定結果
- Supplementary: ResEmoteNet再現手順＋実験ログリンク

---

## 🧠 F. 現状ベースラインまとめ（2025/11時点）
| Model | Macro-F1 | Note |
|--------|-----------|------|
| SVM (Poly) | 0.650 | baseline |
| CNN | 0.6364 | simple conv-based classifier |
| Transformer | 0.6309 | latent sequence-based |
| ResEmoteNet | TBD (Week 2で算出) | SoTA baseline |

---

## ✅ G. 最終目標
> W⁺ 潜在空間を活かした表情分類モデルで、ResEmoteNetを上回る Macro-F1 (>0.70) を達成し、  
> augmentation なし条件下でも統計的有意差を確認する。

---

## 📌 備考
- Fine-tuning した pSp Encoder の更新版は別ブランチで管理（例: `branch: encoder_ft_v2`）  
- augmentation あり/なし条件を分けて記録（例: `experiment_tag: no_aug`, `with_aug`）

---

🧭 **次のステップ:**  
→ Week 1 のタスク（データ前処理統一＋ResEmoteNet環境構築）から開始  
→ その前に「2 の train.py テンプレート」を導入して、改良実験をすぐ走らせられる状態にする
