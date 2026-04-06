#!/usr/bin/env python3
"""
监控FreeSASA批量处理进度
"""

import os
import time
import glob
from datetime import datetime

def monitor_progress():
    """实时监控处理进度"""
    
    output_dir = "alll"
    input_dir = "pdb"
    
    # 获取总文件数
    total_files = len(glob.glob(os.path.join(input_dir, "*.pdb")))
    
    print("FreeSASA Batch Processing Monitor")
    print("=" * 50)
    print(f"Total PDB files: {total_files}")
    print(f"Output directory: {output_dir}")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 50)
    
    try:
        while True:
            # 检查已处理的文件数
            if os.path.exists(output_dir):
                processed_dirs = [d for d in os.listdir(output_dir) 
                                if os.path.isdir(os.path.join(output_dir, d))]
                processed_count = len(processed_dirs)
                
                # 计算进度
                if total_files > 0:
                    progress = (processed_count / total_files) * 100
                else:
                    progress = 0
                
                # 检查失败文件
                fail_file = os.path.join(output_dir, "fail.txt")
                failed_count = 0
                if os.path.exists(fail_file):
                    with open(fail_file, 'r', encoding='utf-8') as f:
                        failed_count = len([line for line in f if not line.startswith('#') and line.strip()])
                
                # 显示进度
                current_time = datetime.now().strftime('%H:%M:%S')
                print(f"\r[{current_time}] Progress: {processed_count}/{total_files} ({progress:.1f}%) | Failed: {failed_count}", end="", flush=True)
                
                # 如果处理完成，显示最终报告
                if processed_count + failed_count >= total_files:
                    print("\n" + "=" * 50)
                    print("Processing completed!")
                    
                    # 显示最终报告
                    report_file = os.path.join(output_dir, "processing_report.txt")
                    if os.path.exists(report_file):
                        print("\nFinal Report:")
                        with open(report_file, 'r', encoding='utf-8') as f:
                            print(f.read())
                    
                    break
            else:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Waiting for processing to start...", end="", flush=True)
            
            time.sleep(5)  # 每5秒更新一次
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def show_detailed_status():
    """显示详细状态"""
    
    output_dir = "alll"
    input_dir = "pdb"
    
    print("Detailed Processing Status")
    print("=" * 50)
    
    # 总文件数
    total_files = len(glob.glob(os.path.join(input_dir, "*.pdb")))
    print(f"Total PDB files: {total_files}")
    
    # 已处理文件
    if os.path.exists(output_dir):
        processed_dirs = [d for d in os.listdir(output_dir) 
                        if os.path.isdir(os.path.join(output_dir, d))]
        processed_count = len(processed_dirs)
        print(f"Processed files: {processed_count}")
        
        # 失败文件
        fail_file = os.path.join(output_dir, "fail.txt")
        if os.path.exists(fail_file):
            with open(fail_file, 'r', encoding='utf-8') as f:
                failed_files = [line.strip() for line in f if not line.startswith('#') and line.strip()]
            print(f"Failed files: {len(failed_files)}")
            
            if failed_files:
                print("\nFailed file IDs:")
                for failed_id in failed_files[:10]:  # 显示前10个
                    print(f"  - {failed_id}")
                if len(failed_files) > 10:
                    print(f"  ... and {len(failed_files) - 10} more")
        else:
            print("Failed files: 0")
        
        # 最近处理的文件
        if processed_dirs:
            print(f"\nRecently processed files:")
            recent_dirs = sorted(processed_dirs)[-10:]  # 最近10个
            for dir_name in recent_dirs:
                dir_path = os.path.join(output_dir, dir_name)
                if os.path.exists(dir_path):
                    # 检查文件修改时间
                    mtime = os.path.getmtime(dir_path)
                    mtime_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
                    print(f"  - {dir_name} ({mtime_str})")
        
        # 处理报告
        report_file = os.path.join(output_dir, "processing_report.txt")
        if os.path.exists(report_file):
            print(f"\nProcessing report exists: {report_file}")
            mtime = os.path.getmtime(report_file)
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Last updated: {mtime_str}")
    else:
        print("Output directory not found - processing not started")

def main():
    """主函数"""
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "watch":
            monitor_progress()
        elif sys.argv[1] == "status":
            show_detailed_status()
        else:
            print("Usage:")
            print("  python monitor_progress.py watch   - Real-time monitoring")
            print("  python monitor_progress.py status  - Show detailed status")
    else:
        print("FreeSASA Processing Monitor")
        print("=" * 30)
        print("Commands:")
        print("  watch  - Real-time monitoring")
        print("  status - Show detailed status")
        print("\nExample:")
        print("  python monitor_progress.py watch")

if __name__ == "__main__":
    main()
