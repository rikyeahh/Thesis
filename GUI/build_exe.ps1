# run with: powershell -ExecutionPolicy ByPass -File "c:\Users\ricca\Desktop\Thesis\GUI\build_exe.ps1"
if (Test-Path .\pyevn\Scripts\activate) {
    echo "venv already present"
} else {
    echo "creating python venv..."
    python3 -m venv pyevn
}

.\pyevn\Scripts\activate
pip3 install -r requirements.txt
rm ./build
rm ./dist
rm ./GUI.spec
pyinstaller .\GUI.py --onefile
