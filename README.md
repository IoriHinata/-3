# Charlie for Android

This repository is prepared to produce an Android APK from `main.py` using
[Buildozer](https://buildozer.readthedocs.io/) and python-for-android.  The
Android manifest receives the `CAMERA` and `RECORD_AUDIO` permissions from
`buildozer.spec`; the program also requests both permissions at runtime before
starting either sensor.

## Build a debug APK (Linux)

Use Ubuntu or another supported Linux environment.  Buildozer does not build
Android APKs natively on Windows or macOS.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip buildozer
buildozer -v android debug
```

The resulting installable debug APK is written to `bin/`, for example
`bin/charlie-1.0.0-arm64-v8a-debug.apk`.  Install it on a device with Android
7.0 (API 24) or later, then grant Camera and Microphone access in the Android
permission prompt.

The first build downloads the Android SDK, NDK and Python-for-Android sources,
so it requires a working Internet connection and can take a while.  Build
artifacts are deliberately excluded from Git.
