import subprocess
import sys
import os
import time
import glob
import threading
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

def show_progress_bar(current, total, bar_length=50):
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = progress * 100
    return f"[{bar}] {current:4d}/{total} ({percent:5.1f}%)"

def monitor_progress_realtime(output_dir, total_ids, stop_event):
    print("\n" + "="*80)
    print("🚀 FreeSASA Batch Processing - Real-time Progress Monitoring")
    print("="*80)

    last_count = 0
    start_time = time.time()

    while not stop_event.is_set():
        try:
            rsa_files = glob.glob(os.path.join(output_dir, "*_complete.rsa"))
            current_count = len(rsa_files)
            elapsed_time = time.time() - start_time
            if elapsed_time > 0 and current_count > 0:
                speed = current_count / elapsed_time * 60 
                if current_count < total_ids:
                    eta_minutes = (total_ids - current_count) / (current_count / elapsed_time) / 60
                    eta_str = f"ETA: {eta_minutes:.1f}min"
                else:
                    eta_str = "Completed!"
            else:
                speed = 0
                eta_str = "Calculating..."
            fail_file = os.path.join(output_dir, "fail.txt")
            failed_count = 0
            if os.path.exists(fail_file):
                with open(fail_file, 'r', encoding='utf-8') as f:
                    failed_count = len([line for line in f if not line.startswith('#') and line.strip()])
            progress_bar = show_progress_bar(current_count, total_ids)
            current_time = datetime.now().strftime('%H:%M:%S')
            new_files = current_count - last_count
            new_indicator = f" (+{new_files})" if new_files > 0 else ""
            print(f"\r[{current_time}] {progress_bar} | Failed: {failed_count:3d} | Speed: {speed:4.1f}/min | {eta_str}{new_indicator}",
                  end="", flush=True)
            last_count = current_count
            if current_count + failed_count >= total_ids:
                print(f"\n\n🎉 Processing Completed!")
                print(f"   Total: {current_count + failed_count}")
                print(f"   Successful: {current_count}")
                print(f"   Failed: {failed_count}")
                print(f"   Time Taken: {elapsed_time/60:.1f} minutes")
                print(f"   Average Speed: {current_count/elapsed_time*60:.1f} files/minute")
                break
            time.sleep(2) 
        except Exception as e:
            print(f"\nMonitoring Error: {e}")
            break

def run_3714_batch_with_progress():
    base_dir = r"/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest"
    id_file = os.path.join(base_dir, "test.txt")
    pdb_dir = os.path.join(base_dir, "pdb")
    output_dir = os.path.join(base_dir, "freesasa_output")
    threads = 8 
    if not os.path.exists(id_file):
        print(f"❌ Error: ID file '{id_file}' not found")
        return
    if not os.path.exists(pdb_dir):
        print(f"❌ Error: PDB directory '{pdb_dir}' not found")
        return
    with open(id_file, 'r') as f:
        total_ids = len([line for line in f if line.strip()])
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "batch_complete_processor.py",
        "--id-file", id_file,
        "--pdb-dir", pdb_dir,
        "--output-dir", output_dir,
        "--threads", str(threads)
    ]
    print("🚀 Starting FreeSASA Batch Processing...")
    print(f"📁 ID File: {id_file} ({total_ids} IDs)")
    print(f"📁 PDB Directory: {pdb_dir}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"🔧 Number of Threads: {threads}")
    log_file = os.path.join(output_dir, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    try:
        print(f"⚡ Starting Processing Process...")
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
        print(f"✅ Background Process Started (PID: {process.pid})")
        print(f"📝 Log File: {log_file}")
        time.sleep(3)
        if process.poll() is None:
            print(f"✅ Process Running Normally")
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_progress_realtime,
                args=(output_dir, total_ids, stop_event)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            try:
                while process.poll() is None:
                    time.sleep(1)
                stop_event.set()
                monitor_thread.join(timeout=5)
                print(f"\n\n✅ Batch Processing Completed!")
            except KeyboardInterrupt:
                print(f"\n\n⚠️ User Interrupt, Process Will Continue in Background")
                print(f"📝 Log File: {log_file}")
                print(f"🔍 Monitoring Command: python monitor_3714.py watch")
                stop_event.set()
        else:
            print(f"❌ Process Start Failed")
            return_code = process.poll()
            print(f"Return Code: {return_code}")
    except Exception as e:
        print(f"❌ Error Starting Process: {e}")

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "start":
        run_3714_batch_with_progress()
    else:
        print("📖 Usage Instructions:")
        print("  python run_rsa_linux.py start")
if __name__ == "__main__":
    main()