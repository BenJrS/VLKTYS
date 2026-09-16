#!/bin/bash
# ============================================================
#  Script build file .dmg cho ca 2 ung dung (chay tren macOS)
#  Cach dung: mo Terminal tai thu muc BaiTapLon, go:
#     chmod +x build_macos.sh && ./build_macos.sh
# ============================================================
set -e

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

for APP in CauA_GaussSeidel CauB_MonteCarlo; do
  echo ""
  echo "=== Building $APP ==="
  cd "$APP"
  pyinstaller --noconfirm --windowed --name "$APP" main.py
  cd dist
  hdiutil create -volname "$APP" -srcfolder "$APP.app" -ov -format UDZO "$APP.dmg"
  cd ../..
  echo "-> $APP/dist/$APP.dmg"
done

echo ""
echo "Hoan tat! File .dmg nam trong:"
echo "  CauA_GaussSeidel/dist/CauA_GaussSeidel.dmg"
echo "  CauB_MonteCarlo/dist/CauB_MonteCarlo.dmg"
