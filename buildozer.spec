[app]
title = Charlie
package.name = charlie
package.domain = org.charliebrain
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,ttf,json
source.exclude_dirs = .git,.buildozer,bin,venv,__pycache__
version = 1.0.0
requirements = python3,kivy,numpy,pyjnius
orientation = portrait
fullscreen = 0
android.permissions = CAMERA,RECORD_AUDIO
android.api = 35
android.minapi = 24
android.ndk_api = 24
android.archs = arm64-v8a,armeabi-v7a
android.allow_backup = False
android.private_storage = True
# Required for non-interactive GitHub Actions SDK installation.
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
