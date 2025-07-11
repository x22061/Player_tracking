#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11x + Hybrid-SORT による人物検出・追跡システム
Hybrid-SORTは外観特徴量と動き情報を組み合わせた高精度追跡アルゴリズム
動画内の人物を検出・追跡し、バウンディングボックスとIDを描画した動画を出力
frame, id, x_center, y_center, width, height の情報をCSVファイルに保存
実行例:
# 仮想環境をアクティベート
cd /Users/x22061/Desktop/YOLO11
source hybrid_sort_env/bin/activate

# 基本的な使用（Hybrid-SORT）
python3 Hybrid_SORT.py input_video.mp4

# 高信頼度で実行
python3 Hybrid_SORT.py input_video.mp4 --conf 0.7

# 異なるトラッカーを選択
python3 Hybrid_SORT.py input_video.mp4 --tracker simple

"""
import cv2
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
from datetime import datetime
import sys

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator
except ImportError:
    print("エラー: ultralyticsがインストールされていません。")
    print("pip install ultralytics でインストールしてください。")
    sys.exit(1)

try:
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracking_utils.timer import Timer
except ImportError:
    print("警告: YOLOXのBYTETrackerが見つかりません。")
    print("Hybrid-SORTの代替実装を使用します。")
    BYTETracker = None

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    print("警告: scipyが見つかりません。ハンガリアンアルゴリズムの代わりにgreedy matchingを使用します。")
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("警告: PyTorchが見つかりません。外観特徴量の抽出は無効になります。")
    TORCH_AVAILABLE = False

# Kalmanフィルタの実装（動き予測用）
class KalmanFilter:
    """
    カルマンフィルタによる物体の動き予測
    """
    def __init__(self, dt=1.0):
        self.dt = dt
        self.initialized = False
        
        # 状態ベクトル: [x, y, vx, vy, w, h, vw, vh]
        self.state = np.zeros(8)
        
        # 状態遷移行列
        self.F = np.array([
            [1, 0, dt, 0, 0, 0, 0, 0],
            [0, 1, 0, dt, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 観測行列
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        # プロセスノイズ共分散行列
        self.Q = np.eye(8) * 0.1
        
        # 観測ノイズ共分散行列
        self.R = np.eye(4) * 1.0
        
        # 誤差共分散行列
        self.P = np.eye(8) * 100
    
    def predict(self):
        """状態予測"""
        if not self.initialized:
            return None
        
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.get_bbox()
    
    def update(self, bbox):
        """観測による状態更新"""
        x, y, w, h = bbox
        z = np.array([x, y, w, h])
        
        if not self.initialized:
            self.state[:4] = z
            self.state[4:] = 0  # 初期速度は0
            self.initialized = True
        else:
            # カルマンゲインの計算
            K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
            
            # 状態更新
            self.state = self.state + K @ (z - self.H @ self.state)
            self.P = (np.eye(8) - K @ self.H) @ self.P
    
    def get_bbox(self):
        """現在の境界ボックスを取得"""
        if not self.initialized:
            return None
        return self.state[:4]


# 外観特徴量抽出器（簡易版）
class FeatureExtractor:
    """
    簡易的な外観特徴量抽出器
    """
    def __init__(self):
        self.feature_dim = 128
    
    def extract(self, image, bbox):
        """
        画像から特徴量を抽出
        """
        try:
            x1, y1, x2, y2 = [int(x) for x in bbox]
            
            # 境界チェック
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.feature_dim)
            
            # ROIを抽出
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return np.zeros(self.feature_dim)
            
            # 簡易的な特徴量（色ヒストグラム + エッジ情報）
            features = []
            
            # 色ヒストグラム（各チャンネル16ビン）
            if len(roi.shape) == 3:
                for c in range(3):
                    hist = cv2.calcHist([roi], [c], None, [16], [0, 256])
                    features.extend(hist.flatten())
            else:
                hist = cv2.calcHist([roi], [0], None, [48], [0, 256])
                features.extend(hist.flatten())
            
            # エッジ情報
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
            features.extend(edge_hist.flatten())
            
            # 正規化
            features = np.array(features)
            features = features / (np.linalg.norm(features) + 1e-6)
            
            # 指定次元に調整
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            elif len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            
            return features
            
        except Exception as e:
            print(f"特徴量抽出エラー: {e}")
            return np.zeros(self.feature_dim)


# Hybrid-SORTトラッカーの実装
class HybridSORTTracker:
    """
    Hybrid-SORT: 外観特徴量と動き情報を組み合わせた追跡アルゴリズム
    """
    def __init__(self, max_lost=30, min_hits=3, iou_threshold=0.3, 
                 appearance_threshold=0.7, motion_weight=0.4, appearance_weight=0.6):
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        self.motion_weight = motion_weight
        self.appearance_weight = appearance_weight
        
        self.tracks = []
        self.track_id_count = 0
        self.feature_extractor = FeatureExtractor()
        
        # 特徴量の履歴を保存（最新N個）
        self.feature_history_size = 5
    
    def update(self, detections, frame=None):
        """
        検出結果でトラックを更新
        """
        if len(detections) == 0:
            # 検出なしの場合の処理
            for track in self.tracks:
                track['kalman'].predict()
                track['lost_count'] += 1
            
            # lost_count が max_lost を超えたトラックを削除
            self.tracks = [t for t in self.tracks if t['lost_count'] <= self.max_lost]
            return []
        
        # 新しい検出の特徴量を抽出
        detection_features = []
        if frame is not None:
            for det in detections:
                feature = self.feature_extractor.extract(frame, det[:4])
                detection_features.append(feature)
        else:
            detection_features = [np.zeros(self.feature_extractor.feature_dim) for _ in detections]
        
        if len(self.tracks) == 0:
            # 初回検出時の処理
            for i, det in enumerate(detections):
                self._create_new_track(det, detection_features[i])
            return [(t['bbox'], t['id']) for t in self.tracks if t['hit_count'] >= self.min_hits]
        
        # 予測ステップ
        for track in self.tracks:
            predicted = track['kalman'].predict()
            if predicted is not None:
                track['predicted_bbox'] = self._xywh_to_xyxy(predicted)
        
        # コスト行列の計算
        cost_matrix = self._compute_cost_matrix(detections, detection_features)
        
        # マッチング
        matches = self._solve_assignment(cost_matrix)
        
        # マッチング結果の処理
        matched_tracks = set()
        matched_detections = set()
        
        for track_idx, det_idx in matches:
            if cost_matrix[track_idx, det_idx] < 1.0:  # 有効なマッチング
                self._update_track(track_idx, detections[det_idx], detection_features[det_idx])
                matched_tracks.add(track_idx)
                matched_detections.add(det_idx)
        
        # マッチしなかったトラックの処理
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['lost_count'] += 1
        
        # マッチしなかった検出から新しいトラックを作成
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self._create_new_track(det, detection_features[i])
        
        # 古いトラックを削除
        self.tracks = [t for t in self.tracks if t['lost_count'] <= self.max_lost]
        
        # 結果を返す
        results = []
        for track in self.tracks:
            if track['hit_count'] >= self.min_hits:
                results.append((track['bbox'], track['id']))
        
        return results
    
    def _compute_cost_matrix(self, detections, detection_features):
        """
        動きと外観の両方を考慮したコスト行列を計算
        """
        num_tracks = len(self.tracks)
        num_detections = len(detections)
        
        if num_tracks == 0 or num_detections == 0:
            return np.ones((num_tracks, num_detections))
        
        # IOUベースの動きコスト
        motion_costs = np.ones((num_tracks, num_detections))
        
        for i, track in enumerate(self.tracks):
            track_bbox = track.get('predicted_bbox', track['bbox'])
            for j, det in enumerate(detections):
                det_bbox = det[:4]
                iou = self._calculate_iou(track_bbox, det_bbox)
                motion_costs[i, j] = 1.0 - iou
        
        # 外観ベースのコスト
        appearance_costs = np.ones((num_tracks, num_detections))
        
        for i, track in enumerate(self.tracks):
            if 'features' in track and len(track['features']) > 0:
                # 過去の特徴量との類似度を計算
                track_feature = np.mean(track['features'], axis=0)
                for j, det_feature in enumerate(detection_features):
                    similarity = np.dot(track_feature, det_feature) / (
                        np.linalg.norm(track_feature) * np.linalg.norm(det_feature) + 1e-6
                    )
                    appearance_costs[i, j] = 1.0 - similarity
        
        # 重み付き結合
        total_costs = (self.motion_weight * motion_costs + 
                      self.appearance_weight * appearance_costs)
        
        return total_costs
    
    def _solve_assignment(self, cost_matrix):
        """
        割り当て問題を解く
        """
        if cost_matrix.size == 0:
            return []
        
        if SCIPY_AVAILABLE:
            # ハンガリアンアルゴリズム
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            return list(zip(row_indices, col_indices))
        else:
            # Greedyマッチング
            return self._greedy_assignment(cost_matrix)
    
    def _greedy_assignment(self, cost_matrix):
        """
        貪欲法による割り当て
        """
        matches = []
        used_rows = set()
        used_cols = set()
        
        # コストが低い順にソート
        indices = np.unravel_index(np.argsort(cost_matrix, axis=None), cost_matrix.shape)
        
        for i, j in zip(indices[0], indices[1]):
            if i not in used_rows and j not in used_cols and cost_matrix[i, j] < 1.0:
                matches.append((i, j))
                used_rows.add(i)
                used_cols.add(j)
        
        return matches
    
    def _create_new_track(self, detection, feature):
        """
        新しいトラックを作成
        """
        self.track_id_count += 1
        bbox = detection[:4]
        confidence = detection[4] if len(detection) > 4 else 1.0
        
        # カルマンフィルタの初期化
        kalman = KalmanFilter()
        kalman.update(self._xyxy_to_xywh(bbox))
        
        new_track = {
            'id': self.track_id_count,
            'bbox': bbox,
            'confidence': confidence,
            'lost_count': 0,
            'hit_count': 1,
            'kalman': kalman,
            'features': [feature],
            'predicted_bbox': bbox
        }
        
        self.tracks.append(new_track)
    
    def _update_track(self, track_idx, detection, feature):
        """
        既存のトラックを更新
        """
        track = self.tracks[track_idx]
        bbox = detection[:4]
        confidence = detection[4] if len(detection) > 4 else 1.0
        
        # トラック情報の更新
        track['bbox'] = bbox
        track['confidence'] = confidence
        track['lost_count'] = 0
        track['hit_count'] += 1
        
        # カルマンフィルタの更新
        track['kalman'].update(self._xyxy_to_xywh(bbox))
        
        # 特徴量履歴の更新
        track['features'].append(feature)
        if len(track['features']) > self.feature_history_size:
            track['features'].pop(0)
    
    def _calculate_iou(self, box1, box2):
        """
        2つのバウンディングボックスのIOUを計算
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 交差領域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _xyxy_to_xywh(self, bbox):
        """
        (x1, y1, x2, y2) -> (x, y, w, h)
        """
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    
    def _xywh_to_xyxy(self, bbox):
        """
        (x, y, w, h) -> (x1, y1, x2, y2)
        """
        x, y, w, h = bbox
        return [x, y, x + w, y + h]


# シンプルなIOUベースのトラッカーを実装（代替用）
class SimpleIOUTracker:
    """シンプルなIOUベースのトラッカー"""
    
    def __init__(self, max_lost=30, min_hits=3, iou_threshold=0.3):
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0
    
    def update(self, detections):
        """検出結果でトラックを更新"""
        if len(detections) == 0:
            # 検出なしの場合、既存トラックの lost_count を増加
            for track in self.tracks:
                track['lost_count'] += 1
            # lost_count が max_lost を超えたトラックを削除
            self.tracks = [t for t in self.tracks if t['lost_count'] <= self.max_lost]
            return []
        
        # IOUマトリックスを計算
        if len(self.tracks) == 0:
            # 既存トラックがない場合、新しいトラックを作成
            for det in detections:
                self.track_id_count += 1
                new_track = {
                    'id': self.track_id_count,
                    'bbox': det[:4],
                    'confidence': det[4],
                    'lost_count': 0,
                    'hit_count': 1
                }
                self.tracks.append(new_track)
            return [(t['bbox'], t['id']) for t in self.tracks if t['hit_count'] >= self.min_hits]
        
        # IOUマッチング
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track['bbox'], det[:4])
        
        # ハンガリアンアルゴリズムの代わりにシンプルなgreedy matching
        matched_tracks = set()
        matched_detections = set()
        matches = []
        
        while True:
            max_iou = 0
            max_indices = (-1, -1)
            for i in range(len(self.tracks)):
                for j in range(len(detections)):
                    if i not in matched_tracks and j not in matched_detections:
                        if iou_matrix[i, j] > max_iou and iou_matrix[i, j] > self.iou_threshold:
                            max_iou = iou_matrix[i, j]
                            max_indices = (i, j)
            
            if max_indices[0] == -1:
                break
            
            matches.append(max_indices)
            matched_tracks.add(max_indices[0])
            matched_detections.add(max_indices[1])
        
        # マッチしたトラックを更新
        for track_idx, det_idx in matches:
            self.tracks[track_idx]['bbox'] = detections[det_idx][:4]
            self.tracks[track_idx]['confidence'] = detections[det_idx][4]
            self.tracks[track_idx]['lost_count'] = 0
            self.tracks[track_idx]['hit_count'] += 1
        
        # マッチしなかったトラックの lost_count を増加
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['lost_count'] += 1
        
        # マッチしなかった検出から新しいトラックを作成
        for j, det in enumerate(detections):
            if j not in matched_detections:
                self.track_id_count += 1
                new_track = {
                    'id': self.track_id_count,
                    'bbox': det[:4],
                    'confidence': det[4],
                    'lost_count': 0,
                    'hit_count': 1
                }
                self.tracks.append(new_track)
        
        # lost_count が max_lost を超えたトラックを削除
        self.tracks = [t for t in self.tracks if t['lost_count'] <= self.max_lost]
        
        # 十分なhit_countを持つトラックのみ返す
        return [(t['bbox'], t['id']) for t in self.tracks if t['hit_count'] >= self.min_hits]
    
    def _calculate_iou(self, box1, box2):
        """2つのバウンディングボックスのIOUを計算"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 交差領域の座標
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # 交差面積
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 各ボックスの面積
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 和集合面積
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class PersonTracker:
    """YOLO11x + Hybrid-SORT による人物検出・追跡システム"""
    
    def __init__(self, model_path=None, tracker_type='hybrid', conf_threshold=0.5):
        """
        初期化
        
        Args:
            model_path (str): YOLOモデルのパス（Noneの場合は事前学習済みモデルを使用）
            tracker_type (str): トラッカーの種類 ('hybrid', 'simple', 'byte')
            conf_threshold (float): 検出の信頼度閾値
        """
        self.conf_threshold = conf_threshold
        
        # YOLOモデルの初期化
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"カスタムモデルを読み込みました: {model_path}")
            else:
                self.model = YOLO('yolo11x.pt')  # 事前学習済みモデル
                print("YOLO11x事前学習済みモデルを読み込みました")
        except Exception as e:
            print(f"YOLOモデルの読み込みエラー: {e}")
            sys.exit(1)
        
        # トラッカーの初期化
        if tracker_type == 'hybrid':
            self.tracker = HybridSORTTracker(
                max_lost=30,
                min_hits=3,
                iou_threshold=0.3,
                appearance_threshold=0.7,
                motion_weight=0.4,
                appearance_weight=0.6
            )
            self.tracker_type = 'hybrid'
            print("Hybrid-SORTトラッカーを使用します")
        elif tracker_type == 'byte' and BYTETracker is not None:
            # BYTETrackerの設定
            class Args:
                track_thresh = 0.5
                track_buffer = 30
                match_thresh = 0.8
                mot20 = False
                frame_rate = 30
            
            self.tracker = BYTETracker(Args(), frame_rate=30)
            self.tracker_type = 'byte'
            print("BYTETrackerを使用します")
        else:
            self.tracker = SimpleIOUTracker()
            self.tracker_type = 'simple'
            print("SimpleIOUTrackerを使用します")
        
        # 結果保存用
        self.tracking_results = []
    
    def detect_persons(self, frame):
        """フレームから人物を検出"""
        try:
            results = self.model(frame, conf=self.conf_threshold, classes=[0], verbose=False)  # class 0 = person
            
            detections = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    if conf >= self.conf_threshold:
                        detections.append([box[0], box[1], box[2], box[3], conf])
            
            return np.array(detections) if detections else np.empty((0, 5))
        except Exception as e:
            print(f"警告: 人物検出中にエラーが発生しました: {e}")
            return np.empty((0, 5))
    
    def process_video(self, input_video_path, output_video_path=None, output_csv_path=None):
        """
        動画を処理して人物検出・追跡を実行
        
        Args:
            input_video_path (str): 入力動画のパス
            output_video_path (str): 出力動画のパス
            output_csv_path (str): 出力CSVのパス
        """
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けません: {input_video_path}")
            return False
        
        # 動画の情報を取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"動画情報: {width}x{height}, {fps}FPS, {total_frames}フレーム")
        
        # 出力動画の設定
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        self.tracking_results = []
        
        print("処理開始...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 人物検出
            detections = self.detect_persons(frame)
            
            # トラッキング
            if self.tracker_type == 'byte' and len(detections) > 0:
                # BYTETracker用の形式に変換
                det_tlwh = []
                det_conf = []
                for det in detections:
                    x1, y1, x2, y2, conf = det
                    w = x2 - x1
                    h = y2 - y1
                    det_tlwh.append([x1, y1, w, h])
                    det_conf.append(conf)
                
                tracks = self.tracker.update(
                    np.array(det_tlwh), 
                    np.array(det_conf), 
                    (height, width)
                )
                
                tracked_objects = []
                for track in tracks:
                    if track.is_activated():
                        x1, y1, w, h = track.tlwh
                        x2 = x1 + w
                        y2 = y1 + h
                        tracked_objects.append(([x1, y1, x2, y2], track.track_id))
            elif self.tracker_type == 'hybrid':
                # Hybrid-SORT（フレーム画像も渡す）
                tracked_objects = self.tracker.update(detections, frame)
            else:
                # SimpleIOUTracker
                tracked_objects = self.tracker.update(detections)
            
            # 結果をCSV用データに保存
            for bbox, track_id in tracked_objects:
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                self.tracking_results.append({
                    'frame': frame_count,
                    'id': track_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': bbox_width,
                    'height': bbox_height
                })
            
            # フレームに描画
            annotator = Annotator(frame, line_width=2)
            
            for bbox, track_id in tracked_objects:
                x1, y1, x2, y2 = bbox
                
                # バウンディングボックスを描画
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # IDを描画
                label = f'ID: {track_id}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # フレーム番号を描画
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 動画に書き込み
            if output_video_path:
                out.write(frame)
            
            # 進捗表示
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"進捗: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # リソースの解放
        cap.release()
        if output_video_path:
            out.release()
        
        print(f"処理完了: {frame_count}フレーム処理")
        print(f"追跡結果: {len(self.tracking_results)}件")
        
        # CSVファイルに保存
        if output_csv_path and self.tracking_results:
            df = pd.DataFrame(self.tracking_results)
            df.to_csv(output_csv_path, index=False)
            print(f"CSVファイルに保存: {output_csv_path}")
            
            # 統計情報を表示
            unique_ids = df['id'].nunique()
            print(f"検出された人物数: {unique_ids}")
            print(f"総フレーム数: {df['frame'].max()}")
        
        return True
    
    def get_tracking_summary(self):
        """追跡結果の統計情報を取得"""
        if not self.tracking_results:
            return None
        
        df = pd.DataFrame(self.tracking_results)
        
        summary = {
            'total_detections': len(df),
            'unique_persons': df['id'].nunique(),
            'total_frames': df['frame'].max(),
            'avg_detections_per_frame': len(df) / df['frame'].max(),
            'persons_per_frame': df.groupby('frame')['id'].nunique().describe()
        }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='YOLO11x + Hybrid-SORT による人物検出・追跡')
    parser.add_argument('input_video', help='入力動画ファイルのパス')
    parser.add_argument('--output_video', '-o', help='出力動画ファイルのパス')
    parser.add_argument('--output_csv', '-c', help='出力CSVファイルのパス')
    parser.add_argument('--model', '-m', help='YOLOモデルファイルのパス')
    parser.add_argument('--tracker', '-t', choices=['hybrid', 'simple', 'byte'], 
                       default='hybrid', help='トラッカーの種類 (hybrid: Hybrid-SORT, simple: SimpleIOU, byte: ByteTracker)')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='検出の信頼度閾値 (デフォルト: 0.5)')
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.input_video):
        print(f"エラー: 入力動画ファイルが見つかりません: {args.input_video}")
        return
    
    # 出力ファイル名の自動生成
    input_path = Path(args.input_video)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not args.output_video:
        args.output_video = str(input_path.parent / f"{input_path.stem}_hybrid_sort_{timestamp}.mp4")
    
    if not args.output_csv:
        args.output_csv = str(input_path.parent / f"{input_path.stem}_hybrid_sort_{timestamp}.csv")
    
    print("=" * 60)
    print("YOLO11x + Hybrid-SORT 人物検出・追跡システム")
    print("=" * 60)
    print(f"入力動画: {args.input_video}")
    print(f"出力動画: {args.output_video}")
    print(f"出力CSV: {args.output_csv}")
    print(f"使用モデル: {args.model if args.model else 'YOLO11x事前学習済み'}")
    print(f"トラッカー: {args.tracker}")
    print(f"信頼度閾値: {args.conf}")
    print("=" * 60)
    
    # トラッカーの初期化と実行
    tracker = PersonTracker(
        model_path=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf
    )
    
    success = tracker.process_video(
        args.input_video,
        args.output_video,
        args.output_csv
    )
    
    if success:
        # 統計情報の表示
        summary = tracker.get_tracking_summary()
        if summary:
            print("\n" + "=" * 40)
            print("追跡結果統計")
            print("=" * 40)
            print(f"総検出数: {summary['total_detections']:,}")
            print(f"ユニーク人物数: {summary['unique_persons']}")
            print(f"総フレーム数: {summary['total_frames']:,}")
            print(f"フレームあたりの平均検出数: {summary['avg_detections_per_frame']:.2f}")
            
            if args.tracker == 'hybrid':
                print("\nHybrid-SORTの特徴:")
                print("- 外観特徴量と動き情報を組み合わせた高精度追跡")
                print("- カルマンフィルタによる動き予測")
                print("- 色ヒストグラム+エッジ情報による外観マッチング")
        
        print("\n" + "=" * 40)
        print("処理が正常に完了しました。")
        print("=" * 40)
    else:
        print("\nエラー: 処理に失敗しました。")


if __name__ == "__main__":
    main()
