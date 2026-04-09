import os
import subprocess
import sys
import urllib.request
import shutil
import platform
def install_python_dependencies():
    print("📦 Installing Python dependencies...")
    try:
        # Explicitly install numpy first to ensure it is available
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        
        # Install all dependencies from requirements.txt
        if os.path.exists("requirements.txt"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            print("❌ requirements.txt not found.")
            sys.exit(1)
            
        print("✅ Python dependencies installed.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        print("   Note: faster-whisper requires Python 3.10 - 3.12.")
        sys.exit(1)

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"✅ {filename} already exists.")
        return

    print(f"⬇️ Downloading {filename}...")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rDownloading... {percent}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, filename, reporthook)
        print(f"\n✅ {filename} downloaded.")
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")

def check_system_requirements():
    print("🔍 Checking system requirements...")
    system = platform.system()
    
    if system == "Windows":
        # 1. Check Ollama
        print("   [System] Checking Ollama...")
        if shutil.which("ollama") is None:
            print("⚠️ Ollama not found in PATH.")
            print("⬇️ Downloading Ollama installer...")
            download_file("https://ollama.com/download/OllamaSetup.exe", "OllamaSetup.exe")
            print("ℹ️ Please run 'OllamaSetup.exe' to install Ollama, then restart this script.")
        else:
            print("✅ Ollama is installed.")

        # 2. Check eSpeak NG
        print("   [System] Checking eSpeak NG...")
        espeak_path = shutil.which("espeak-ng")
        if espeak_path is None:
            # Check default install location
            possible_paths = [
                r"C:\Program Files\eSpeak NG\espeak-ng.exe",
                r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe"
            ]
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"✅ eSpeak NG found at '{path}' (will be configured in main.py).")
                    found = True
                    break
            
            if not found:
                print("⚠️ eSpeak NG not found in standard locations.")
                print("⬇️ Downloading eSpeak NG installer (Required for Phonemizer)...")
                url = "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi"
                download_file(url, "espeak-ng-X64.msi")
                print("ℹ️ Please run 'espeak-ng-X64.msi' to install eSpeak NG.")
        else:
            print("✅ eSpeak NG is installed and in PATH.")

        # 3. Check FFmpeg
        print("   [System] Checking FFmpeg...")
        if shutil.which("ffmpeg") is None:
            print("⚠️ FFmpeg not found in PATH.")
            print("   Video processing features (dubbing) will be disabled.")
            print("   👉 Install via Winget: winget install Gyan.FFmpeg")
            print("   👉 Or download from: https://ffmpeg.org/download.html")
        else:
            print("✅ FFmpeg is installed.")

    elif system == "Darwin":  # macOS
        print("   [System] Checking Ollama...")
        if shutil.which("ollama") is None:
            print("⚠️ Ollama not found. Please download and install from https://ollama.com/")
        else:
            print("✅ Ollama is installed.")

        print("   [System] Checking eSpeak NG...")
        if shutil.which("espeak-ng") is None:
            print("⚠️ eSpeak NG not found. 👉 Install via Homebrew: brew install espeak-ng")
        else:
            print("✅ eSpeak NG is installed.")

        print("   [System] Checking FFmpeg...")
        if shutil.which("ffmpeg") is None:
            print("⚠️ FFmpeg not found. 👉 Install via Homebrew: brew install ffmpeg")
        else:
            print("✅ FFmpeg is installed.")

    elif system == "Linux":
        # Generic checks for Linux, assuming apt package manager for suggestions
        for pkg, install_cmd in [("ollama", "See installation instructions at https://ollama.com/"), ("espeak-ng", "sudo apt-get install espeak-ng"), ("ffmpeg", "sudo apt-get install ffmpeg")]:
            print(f"   [System] Checking {pkg}...")
            if shutil.which(pkg) is None:
                print(f"⚠️ {pkg} not found. 👉 {install_cmd}")
            else:
                print(f"✅ {pkg} is installed.")

def pull_ollama_model(model_name="llama3.2:3b"):
    print(f"🦙 Pulling Ollama model ({model_name})...")
    try:
        # Check if ollama is installed
        subprocess.check_call(["ollama", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["ollama", "pull", model_name])
        subprocess.check_call(["ollama", "pull", "llama3.2:1b"])
        subprocess.check_call(["ollama", "pull", "llama3.1:8b"])
        print("✅ Ollama model pulled.")
    except FileNotFoundError:
        print("❌ Ollama is not installed or not in PATH. Please install Ollama from https://ollama.com/")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull Ollama model: {e}")

def main():
    print("🚀 Starting Vocal Agent Setup...")
    
    # 1. Check System Requirements (Download installers if needed)
    check_system_requirements()

    # 2. Install Python Dependencies
    install_python_dependencies()

    # 3. Download AI Models (Kokoro)
    # URLs derived from README: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
    models = [
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", "kokoro-v1.0.onnx"),
        ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", "voices-v1.0.bin")
    ]
    
    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")

    for url, filename in models:
        target_path = os.path.join("models", filename)
        download_file(url, target_path)

    # 4. Pull Ollama Model
    pull_ollama_model("llama3.2:3b")

    print("\n🎉 Setup actions completed.")
    print("ℹ️  If installers were downloaded (OllamaSetup.exe, espeak-ng-X64.msi), please run them manually.")

if __name__ == "__main__":
    main()