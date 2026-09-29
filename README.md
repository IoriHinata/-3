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

## Сборка APK прямо в GitHub Actions

В репозитории есть готовый workflow
[`.github/workflows/android-apk.yml`](.github/workflows/android-apk.yml). Он
собирает **debug APK** на облачном Linux-раннере GitHub; локально устанавливать
Android SDK, NDK, Java или Buildozer не требуется.

1. Создайте репозиторий на GitHub и загрузите в него все файлы этого проекта,
   включая `main.py`, `buildozer.spec` и каталог `.github`.
2. Откройте вкладку **Actions** репозитория. Если GitHub предложит включить
   workflow, подтвердите **I understand my workflows, go ahead and enable
   them**.
3. В левой панели выберите **Build Android APK**, нажмите **Run workflow**,
   оставьте ветку с проектом и подтвердите **Run workflow**.
4. Дождитесь успешного статуса с зелёной галочкой у задания **Build debug
   APK**. Первая сборка обычно самая долгая: Buildozer загружает Android SDK,
   NDK и зависимости.
5. Откройте завершившийся запуск, в блоке **Artifacts** скачайте
   `charlie-android-apk`. Распакуйте ZIP: внутри находится файл `*.apk`.
6. Передайте APK на телефон, разрешите установку из выбранного источника в
   Android и установите его. При первом нажатии «СТАРТ» разрешите доступ к
   камере и микрофону.

Workflow запускает официальный Docker-образ `kivy/buildozer` напрямую, а не
устаревший сторонний GitHub Action. Это важно: ошибка `chown: invalid user:
'user'` означает проблему такого стороннего action ещё до запуска Buildozer;
в текущем workflow эта причина устранена.

Если workflow завершается ошибкой, откройте шаг **Build with Buildozer** и
скопируйте его журнал: в нём находится точная причина. Артефакт хранится 14
дней, после чего для получения APK нужно запустить workflow снова.
