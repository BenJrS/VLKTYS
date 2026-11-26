#!/bin/bash
# Build 2 ứng dụng cho macOS (.app + .dmg). Chạy tại thư mục gốc:  ./build_macos.sh
set -e
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
for APP in CauA_GaussSeidel CauB_MonteCarlo; do
  echo "=== Build $APP ==="
  (cd "$APP" && pyinstaller --noconfirm "$APP.spec" \
     && hdiutil create -volname "$APP" -srcfolder "dist/$APP.app" -ov -format UDZO "dist/$APP.dmg")
done
echo "Xong: CauA_GaussSeidel/dist/CauA_GaussSeidel.dmg, CauB_MonteCarlo/dist/CauB_MonteCarlo.dmg"
