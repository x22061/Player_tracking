# YOLO11x + Hybrid-SORT 人物検出・追跡システム

このシステムは、YOLO11xとHybrid-SORTアルゴリズムを組み合わせて、動画内の人物を高精度で検出・追跡し、バウンディングボックスとIDを描画した動画を出力するとともに、追跡データをCSVファイルに保存します。

## 機能

- **人物検出**: YOLO11xを使用した高精度な人物検出
- **人物追跡**: Hybrid-SORTによる外観特徴量と動き情報を組み合わせた高性能追跡
- **動画出力**: バウンディングボックスとIDが描画された動画ファイルの生成
- **データ保存**: frame, id, x_center, y_center, width, height の情報をCSVファイルに保存
- **統計情報**: 追跡結果の統計情報を表示

## Hybrid-SORTの特徴

Hybrid-SORTは以下の技術を組み合わせた最先端の追跡アルゴリズムです：

1. **カルマンフィルタ**: 物体の動きを予測し、フレーム間での位置推定を改善
2. **外観特徴量**: 色ヒストグラムとエッジ情報を組み合わせた外観マッチング
3. **IOUマッチング**: 境界ボックスの重複度による位置ベースのマッチング
4. **ハンガリアンアルゴリズム**: 最適な検出-トラック割り当て（scipyが利用可能な場合）
5. **適応的閾値**: 動きと外観の重み付けによる柔軟なマッチング

## 必要な環境

### Python環境
- Python 3.8以上

### 依存ライブラリ
以下のコマンドで必要なライブラリをインストールしてください：

```bash
pip install -r requirements.txt
```

### 主要な依存関係
- ultralytics (YOLO11x)
- opencv-python (動画処理)
- pandas (データ処理)
- numpy (数値計算)
- scipy (科学計算、ハンガリアンアルゴリズム)
- torch, torchvision (深層学習)

## 使用方法

### 基本的な使用方法（Hybrid-SORTを使用）

```bash
python Hybrid-SORT.py input_video.mp4
```

### オプション付きの使用方法

```bash
python Hybrid-SORT.py input_video.mp4 \
    --output_video output_tracked.mp4 \
    --output_csv tracking_data.csv \
    --model custom_model.pt \
    --tracker hybrid \
    --conf 0.6
```

### パラメータ説明

- `input_video`: 入力動画ファイルのパス（必須）
- `--output_video, -o`: 出力動画ファイルのパス（省略時は自動生成）
- `--output_csv, -c`: 出力CSVファイルのパス（省略時は自動生成）
- `--model, -m`: カスタムYOLOモデルのパス（省略時は事前学習済みYOLO11xを使用）
- `--tracker, -t`: トラッカーの種類（`hybrid`, `simple`, `byte`、デフォルト: `hybrid`）
- `--conf`: 検出の信頼度閾値（デフォルト: 0.5）

### トラッカーの種類

- **hybrid**: Hybrid-SORT（推奨）- 外観特徴量と動き情報を組み合わせた高精度追跡
- **simple**: SimpleIOUTracker - シンプルなIOUベースの追跡
- **byte**: BYTETracker - YOLOXのBYTETracker（yoloxライブラリが必要）

### 使用例

#### 1. 基本的な人物追跡（Hybrid-SORT使用）
```bash
python Hybrid-SORT.py sample_video.mp4
```

#### 2. 高い信頼度でHybrid-SORT追跡
```bash
python Hybrid-SORT.py sample_video.mp4 --conf 0.7
```

#### 3. カスタムモデルでHybrid-SORT使用
```bash
python Hybrid-SORT.py sample_video.mp4 --model my_custom_yolo.pt
```

#### 4. 出力ファイル名を指定してHybrid-SORT実行
```bash
python Hybrid-SORT.py sample_video.mp4 \
    --output_video tracked_result.mp4 \
    --output_csv tracking_result.csv
```

#### 5. 異なるトラッカーを選択
```bash
# Hybrid-SORT（推奨）
python Hybrid-SORT.py sample_video.mp4 --tracker hybrid

# シンプルなIOUトラッカー
python Hybrid-SORT.py sample_video.mp4 --tracker simple

# BYTETracker（yoloxライブラリが必要）
python Hybrid-SORT.py sample_video.mp4 --tracker byte
```

## 出力ファイル

### 動画ファイル
- バウンディングボックス（緑色の矩形）
- 人物ID（各バウンディングボックスの上部に表示）
- フレーム番号（左上に表示）

### CSVファイル
以下の列を含むCSVファイルが生成されます：

| 列名 | 説明 |
|------|------|
| frame | フレーム番号 |
| id | 人物ID |
| x_center | バウンディングボックスの中心X座標 |
| y_center | バウンディングボックスの中心Y座標 |
| width | バウンディングボックスの幅 |
| height | バウンディングボックスの高さ |

## システム構成

### PersonTracker クラス
メインの追跡システムクラス：
- YOLO11xモデルの初期化
- トラッカーの初期化
- 動画処理とデータ保存

### SimpleIOUTracker クラス
IOUベースのシンプルなトラッカー：
- IOUマッチングによる追跡
- 新しいトラックの作成
- 失われたトラックの管理

## トラッカーの選択

### SimpleIOUTracker（デフォルト）
- **利点**: 追加の依存関係不要、軽量
- **適用場面**: 比較的シンプルなシーン、少数の人物

### BYTETracker（オプション）
- **利点**: より高精度な追跡
- **必要**: YOLOXライブラリのインストール
- **適用場面**: 複雑なシーン、多数の人物

## トラブルシューティング

### よくある問題

1. **YOLOモデルのダウンロードエラー**
   ```bash
   # インターネット接続を確認してください
   # 初回実行時にYOLO11xモデルが自動ダウンロードされます
   ```

2. **GPU使用時のメモリエラー**
   ```bash
   # CPUモードで実行するか、バッチサイズを小さくしてください
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **BYTETrackerが利用できない**
   ```bash
   # YOLOXをインストールしてください
   pip install yolox
   ```

## パフォーマンス最適化

### GPU使用
CUDA対応GPUがある場合、自動的にGPUが使用されます。

### 処理速度向上
- 信頼度閾値を上げる（`--conf 0.7`など）
- より軽量なYOLOモデルを使用（yolo11s.ptなど）
- フレームレートを下げる（事前に動画を変換）

## ライセンス

このプロジェクトは研究目的で作成されています。
使用する際は、各ライブラリのライセンスを確認してください。

## 参考文献

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)
