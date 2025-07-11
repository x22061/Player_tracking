#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid-SORT.py の使用例とテストスクリプト
YOLO11x + Hybrid-SORT による人物検出・追跡の実行例
"""

import os
import sys
from pathlib import Path
import subprocess

def run_example():
    """使用例の実行"""
    
    print("=== YOLO11x + Hybrid-SORT 使用例 ===\n")
    
    # 現在のディレクトリ
    current_dir = Path(__file__).parent
    
    # サンプル動画ファイルの確認
    sample_videos = [
        "input_video.mp4",
        "sample.mp4",
        "test.mp4",
        "input.mp4"
    ]
    
    input_video = None
    for video in sample_videos:
        video_path = current_dir / video
        if video_path.exists():
            input_video = str(video_path)
            break
    
    if not input_video:
        print("注意: サンプル動画ファイルが見つかりません。")
        print("以下のいずれかのファイル名で動画を配置してください:")
        for video in sample_videos:
            print(f"  - {video}")
        print("\n手動でファイルパスを指定して実行することも可能です:")
        print("python Hybrid-SORT.py /path/to/your/video.mp4")
        return
    
    print(f"サンプル動画を発見: {input_video}")
    
    # 各種実行例
    examples = [
        {
            "name": "基本的な実行例（Hybrid-SORT）",
            "description": "Hybrid-SORTで人物検出・追跡を実行",
            "command": ["python", "Hybrid-SORT.py", input_video]
        },
        {
            "name": "高信頼度での実行例（Hybrid-SORT）",
            "description": "信頼度閾値を0.7に設定してHybrid-SORTで実行",
            "command": ["python", "Hybrid-SORT.py", input_video, "--conf", "0.7"]
        },
        {
            "name": "SimpleIOUTrackerでの実行例",
            "description": "シンプルなIOUベーストラッカーで実行",
            "command": ["python", "Hybrid-SORT.py", input_video, "--tracker", "simple"]
        },
        {
            "name": "出力ファイル指定例（Hybrid-SORT）",
            "description": "出力動画とCSVファイル名を指定してHybrid-SORTで実行",
            "command": [
                "python", "Hybrid-SORT.py", input_video,
                "--output_video", "hybrid_tracked_output.mp4",
                "--output_csv", "hybrid_tracking_data.csv",
                "--tracker", "hybrid"
            ]
        },
        {
            "name": "全トラッカー比較実行",
            "description": "3つのトラッカーで同じ動画を処理して比較",
            "command": "comparison"  # 特別なフラグ
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print(f"   説明: {example['description']}")
        
        if example['command'] == "comparison":
            print(f"   この機能では3つのトラッカーで同じ動画を処理します:")
            print(f"   - Hybrid-SORT")
            print(f"   - SimpleIOUTracker") 
            print(f"   - BYTETracker (利用可能な場合)")
        else:
            print(f"   コマンド: {' '.join(example['command'])}")
        
        response = input(f"   この例を実行しますか？ (y/n): ").lower().strip()
        
        if response == 'y':
            if example['command'] == "comparison":
                run_tracker_comparison(input_video, current_dir)
            else:
                print(f"   実行中...")
                try:
                    result = subprocess.run(example['command'], 
                                          capture_output=True, 
                                          text=True, 
                                          cwd=current_dir)
                    
                    if result.returncode == 0:
                        print(f"   ✓ 実行成功")
                        if result.stdout:
                            print(f"   出力: {result.stdout}")
                    else:
                        print(f"   ✗ 実行失敗")
                        if result.stderr:
                            print(f"   エラー: {result.stderr}")
                            
                except Exception as e:
                    print(f"   ✗ 実行エラー: {e}")
        else:
            print(f"   スキップしました")
    
    print("\n=== 実行例終了 ===")
    
    # 追加情報
    print("\n=== 追加情報 ===")
    print("1. 必要な依存関係のインストール:")
    print("   pip install -r requirements.txt")
    print("\n2. ヘルプの表示:")
    print("   python YOLOx_IOF.py --help")
    print("\n3. 出力ファイルの確認:")
    print("   - 追跡動画: *_tracked_*.mp4")
    print("   - 追跡データ: *_tracking_*.csv")

def check_dependencies():
    """依存関係の確認"""
    print("=== 依存関係チェック ===\n")
    
    required_packages = [
        "ultralytics",
        "opencv-python", 
        "pandas",
        "numpy",
        "torch"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package} - インストール済み")
        except ImportError:
            print(f"✗ {package} - 未インストール")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n以下のパッケージをインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n全ての依存関係が満たされています。")
        return True

def run_tracker_comparison(input_video, current_dir):
    """複数のトラッカーで同じ動画を処理して比較"""
    print("\n   === トラッカー比較実行 ===")
    
    trackers = [
        ("hybrid", "Hybrid-SORT"),
        ("simple", "SimpleIOUTracker"),
        ("byte", "BYTETracker")
    ]
    
    results = {}
    
    for tracker_type, tracker_name in trackers:
        print(f"\n   {tracker_name}で処理中...")
        
        command = [
            "python", "Hybrid-SORT.py", input_video,
            "--tracker", tracker_type,
            "--output_video", f"comparison_{tracker_type}.mp4",
            "--output_csv", f"comparison_{tracker_type}.csv"
        ]
        
        try:
            import time
            start_time = time.time()
            
            result = subprocess.run(command, 
                                  capture_output=True, 
                                  text=True, 
                                  cwd=current_dir)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"   ✓ {tracker_name} 処理成功 ({processing_time:.1f}秒)")
                results[tracker_type] = {
                    "success": True,
                    "time": processing_time,
                    "output": result.stdout
                }
            else:
                print(f"   ✗ {tracker_name} 処理失敗")
                if result.stderr:
                    print(f"     エラー: {result.stderr}")
                results[tracker_type] = {
                    "success": False,
                    "time": processing_time,
                    "error": result.stderr
                }
                
        except Exception as e:
            print(f"   ✗ {tracker_name} 実行エラー: {e}")
            results[tracker_type] = {
                "success": False,
                "time": 0,
                "error": str(e)
            }
    
    # 結果サマリー
    print(f"\n   === 比較結果サマリー ===")
    for tracker_type, tracker_name in trackers:
        if tracker_type in results:
            result = results[tracker_type]
            if result["success"]:
                print(f"   {tracker_name:<15}: ✓ 成功 ({result['time']:.1f}秒)")
            else:
                print(f"   {tracker_name:<15}: ✗ 失敗")
    
    # 出力ファイルの確認
    print(f"\n   生成されたファイル:")
    for tracker_type, tracker_name in trackers:
        if tracker_type in results and results[tracker_type]["success"]:
            video_file = current_dir / f"comparison_{tracker_type}.mp4"
            csv_file = current_dir / f"comparison_{tracker_type}.csv"
            if video_file.exists():
                print(f"   - {video_file.name} ({tracker_name})")
            if csv_file.exists():
                print(f"   - {csv_file.name} ({tracker_name})")

def main():
    """メイン関数"""
    print("YOLO11x + IOF-Tracker テストスクリプト\n")
    
    # 依存関係チェック
    if not check_dependencies():
        print("\n依存関係を解決してから再実行してください。")
        return
    
    print("\n" + "="*50)
    
    # 使用例の実行
    run_example()

if __name__ == "__main__":
    main()
