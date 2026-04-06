#!/usr/bin/env python3
####################批量生成freesasa结果###################
####################路径为英文############################
"""
运行3714批量处理 - 带进度条的增强版本
"""

import subprocess
import sys
import os
import time
import glob
import threading
from datetime import datetime

def show_progress_bar(current, total, bar_length=50):
    """显示进度条"""
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = progress * 100
    return f"[{bar}] {current:4d}/{total} ({percent:5.1f}%)"

def monitor_progress_realtime(output_dir, total_ids, stop_event):
    """实时监控进度"""
    print("\n" + "="*80)
    print("🚀 FreeSASA 18660 批量处理 - 实时进度监控")
    print("="*80)

    last_count = 0
    start_time = time.time()

    while not stop_event.is_set():
        try:
            # 统计已处理的文件
            rsa_files = glob.glob(os.path.join(output_dir, "*_complete.rsa"))
            current_count = len(rsa_files)

            # 计算速度和ETA
            elapsed_time = time.time() - start_time
            if elapsed_time > 0 and current_count > 0:
                speed = current_count / elapsed_time * 60  # 文件/分钟
                if current_count < total_ids:
                    eta_minutes = (total_ids - current_count) / (current_count / elapsed_time) / 60
                    eta_str = f"ETA: {eta_minutes:.1f}min"
                else:
                    eta_str = "完成!"
            else:
                speed = 0
                eta_str = "计算中..."

            # 检查失败文件
            fail_file = os.path.join(output_dir, "fail.txt")
            failed_count = 0
            if os.path.exists(fail_file):
                with open(fail_file, 'r', encoding='utf-8') as f:
                    failed_count = len([line for line in f if not line.startswith('#') and line.strip()])

            # 显示进度
            progress_bar = show_progress_bar(current_count, total_ids)
            current_time = datetime.now().strftime('%H:%M:%S')
            new_files = current_count - last_count
            new_indicator = f" (+{new_files})" if new_files > 0 else ""

            # 清除当前行并显示新进度
            print(f"\r[{current_time}] {progress_bar} | 失败: {failed_count:3d} | 速度: {speed:4.1f}/min | {eta_str}{new_indicator}",
                  end="", flush=True)

            last_count = current_count

            # 如果处理完成
            if current_count + failed_count >= total_ids:
                print(f"\n\n🎉 处理完成!")
                print(f"   总计: {current_count + failed_count}")
                print(f"   成功: {current_count}")
                print(f"   失败: {failed_count}")
                print(f"   用时: {elapsed_time/60:.1f} 分钟")
                print(f"   平均速度: {current_count/elapsed_time*60:.1f} 文件/分钟")
                break

            time.sleep(2)  # 每2秒更新一次

        except Exception as e:
            print(f"\n监控出错: {e}")
            break

def run_3714_batch_with_progress():
    """运行3714批量处理（带进度条）"""

    # 使用英文路径避免编码问题
    base_dir = r"E:"
    id_file = os.path.join(base_dir, "sar8170.txt")
    pdb_dir = os.path.join(base_dir, "pdb")
    output_dir = os.path.join(base_dir, "freesasa")

    threads = 8  # 使用8个线程

    # 检查输入文件
    if not os.path.exists(id_file):
        print(f"❌ Error: ID file '{id_file}' not found")
        return

    if not os.path.exists(pdb_dir):
        print(f"❌ Error: PDB directory '{pdb_dir}' not found")
        return

    # 读取ID数量
    with open(id_file, 'r') as f:
        total_ids = len([line for line in f if line.strip()])

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建命令
    cmd = [
        sys.executable,
        "batch_complete_processor.py",
        "--id-file", id_file,
        "--pdb-dir", pdb_dir,
        "--output-dir", output_dir,
        "--threads", str(threads)
    ]

    print("🚀 启动 FreeSASA 18660 批量处理...")
    print(f"📁 ID文件: {id_file} ({total_ids} IDs)")
    print(f"📁 PDB目录: {pdb_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔧 线程数: {threads}")

    # 创建日志文件
    log_file = os.path.join(output_dir, f"batch_18660_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    try:
        # 启动处理进程
        print(f"⚡ 启动处理进程...")

        if os.name == 'nt':  # Windows
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
        else:  # Unix/Linux
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )

        print(f"✅ 后台进程已启动 (PID: {process.pid})")
        print(f"📝 日志文件: {log_file}")

        # 等待进程启动
        time.sleep(3)

        # 检查进程是否还在运行
        if process.poll() is None:
            print(f"✅ 进程运行正常")

            # 启动实时进度监控
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_progress_realtime,
                args=(output_dir, total_ids, stop_event)
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            try:
                # 等待用户中断或进程完成
                while process.poll() is None:
                    time.sleep(1)

                # 进程完成，停止监控
                stop_event.set()
                monitor_thread.join(timeout=5)

                print(f"\n\n✅ 批量处理已完成!")

            except KeyboardInterrupt:
                print(f"\n\n⚠️  用户中断，进程将继续在后台运行")
                print(f"📝 日志文件: {log_file}")
                print(f"🔍 监控命令: python monitor_3714.py watch")
                stop_event.set()

        else:
            print(f"❌ 进程启动失败")
            return_code = process.poll()
            print(f"返回码: {return_code}")

    except Exception as e:
        print(f"❌ 启动进程时出错: {e}")

def run_3714_batch_background():
    """后台运行3714批量处理（无进度条）"""
    # 使用英文路径避免编码问题
    base_dir = r"D:\Pycharm\PyCharmProjects"
    id_file = os.path.join(base_dir, "17363.txt")
    pdb_dir = os.path.join(base_dir, "17363pdb")
    output_dir = os.path.join(base_dir, "17363freesasa")
    # base_dir = r"E:\PPI-predict\5562\5314"
    # id_file = os.path.join(base_dir, "5314.txt")
    # pdb_dir = os.path.join(base_dir, "5314pdb")
    # output_dir = os.path.join(base_dir, "531freesasa")
    threads = 1  # 使用8个线程

    # id_file = "18660.txt"
    # pdb_dir = "pdb"
    # output_dir = "3714"
    # threads = 8

    # 检查输入文件
    if not os.path.exists(id_file):
        print(f"❌ Error: ID file '{id_file}' not found")
        return

    if not os.path.exists(pdb_dir):
        print(f"❌ Error: PDB directory '{pdb_dir}' not found")
        return

    # 读取ID数量
    with open(id_file, 'r') as f:
        total_ids = len([line for line in f if line.strip()])

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 构建命令
    cmd = [
        sys.executable,
        "batch_complete_processor.py",
        "--id-file", id_file,
        "--pdb-dir", pdb_dir,
        "--output-dir", output_dir,
        "--threads", str(threads)
    ]

    print("🚀 启动 FreeSASA 18660 后台批量处理...")
    print(f"📁 ID文件: {id_file} ({total_ids} IDs)")
    print(f"📁 PDB目录: {pdb_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔧 线程数: {threads}")

    # 创建日志文件
    log_file = os.path.join(output_dir, f"batch_18660_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    try:
        # 在Windows上后台运行
        if os.name == 'nt':  # Windows
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                )
            print(f"✅ 后台进程已启动 (PID: {process.pid})")
        else:  # Unix/Linux
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
            print(f"✅ 后台进程已启动 (PID: {process.pid})")

        print(f"📝 日志文件: {log_file}")
        print(f"🔍 进度监控: python monitor_3714.py watch")
        print(f"📊 快速状态: python quick_status.py")
        print(f"⏱️  预估时间: ~{total_ids * 0.6 / threads:.1f} 分钟")

        # 等待几秒钟确保进程启动
        time.sleep(3)

        # 检查进程是否还在运行
        if process.poll() is None:
            print("✅ 进程运行正常，已在后台执行")
        else:
            print("❌ 进程启动失败")
            return_code = process.poll()
            print(f"返回码: {return_code}")

    except Exception as e:
        print(f"❌ 启动后台进程时出错: {e}")

def check_3714_progress():
    """检查3714处理进度"""

    # 使用英文路径避免编码问题
    output_dir = r"D:\Pycharm\PyCharmProjects\18660\PPI\18660freesasa"

    if not os.path.exists(output_dir):
        print("❌ 处理尚未开始 - 输出目录不存在")
        return

    # 统计已生成的文件
    rsa_files = glob.glob(os.path.join(output_dir, "*_complete.rsa"))
    atoms_files = glob.glob(os.path.join(output_dir, "*_atoms_complete.txt"))
    residues_files = glob.glob(os.path.join(output_dir, "*_residues_complete.txt"))
    stats_files = glob.glob(os.path.join(output_dir, "*_statistics.txt"))

    processed_ids = len(rsa_files)  # 以RSA文件数量为准

    print("📊 18660 批量处理状态")
    print("=" * 40)
    print(f"总计ID: 18660")
    print(f"已处理: {processed_ids}")
    print(f"进度: {processed_ids/18660*100:.1f}%")
    print(f"剩余: {18660 - processed_ids}")

    print(f"\n📁 生成的文件:")
    print(f"  RSA文件: {len(rsa_files)}")
    print(f"  原子文件: {len(atoms_files)}")
    print(f"  残基文件: {len(residues_files)}")
    print(f"  统计文件: {len(stats_files)}")

    # 检查失败文件
    fail_file = os.path.join(output_dir, "fail.txt")
    if os.path.exists(fail_file):
        with open(fail_file, 'r', encoding='utf-8') as f:
            failed_ids = [line.strip() for line in f if not line.startswith('#') and line.strip()]
        print(f"  失败ID: {len(failed_ids)}")

        if failed_ids:
            print(f"  最近失败: {failed_ids[-5:]}")
    else:
        print(f"  失败ID: 0")

    # 检查最新处理的文件
    if rsa_files:
        latest_file = max(rsa_files, key=os.path.getmtime)
        latest_time = os.path.getmtime(latest_file)
        latest_time_str = datetime.fromtimestamp(latest_time).strftime('%H:%M:%S')
        latest_id = os.path.basename(latest_file).replace('_complete.rsa', '')

        print(f"\n⏰ 最新处理: {latest_id} ({latest_time_str})")

        # 检查是否还在处理
        if time.time() - latest_time < 300:  # 5分钟内
            print(f"状态: ✅ 正在处理")
        else:
            print(f"状态: ⚠️  可能已停止")

def main():
    """主函数"""

    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            run_3714_batch_with_progress()
        elif sys.argv[1] == "background" or sys.argv[1] == "bg":
            run_3714_batch_background()
        elif sys.argv[1] == "status":
            check_3714_progress()
        elif sys.argv[1] == "help":
            print("📖 使用说明:")
            print("  python run_3714_batch.py start      - 启动处理（带进度条）")
            print("  python run_3714_batch.py background - 后台运行（无进度条）")
            print("  python run_3714_batch.py bg         - 后台运行（简写）")
            print("  python run_3714_batch.py status     - 检查处理状态")
            print("  python run_3714_batch.py help       - 显示帮助")
        else:
            print("❌ 未知命令。使用 'help' 查看使用说明。")
    else:
        print("🚀 FreeSASA 18660 批量处理器")
        print("=" * 35)
        print("📖 命令:")
        print("  start      - 启动处理（带实时进度条）")
        print("  background - 后台运行（无进度条）")
        print("  status     - 检查处理状态")
        print("  help       - 显示帮助")
        print("\n💡 示例:")
        print("  python run_3714_batch.py start")
        print("  python run_3714_batch.py background")

if __name__ == "__main__":
    main()


####################批量生成freesasa结果###################
####################路径为英文############################