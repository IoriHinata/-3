# -*- coding: utf-8 -*-
"""
CHARLIE — developmental multimodal brain
Android / Pydroid 3 / Kivy

No built-in words, questions or sentences.
The initial vocal system contains only primitive phonetic exploration sounds.
The system learns acoustic units, visual prototypes, cross-modal associations,
repeated sequences and internal communication pressure.

Files:
    charlie_final_android.py
    charlie_brain.json  (created automatically)

Recommended Pydroid packages:
    numpy
    kivy
    pyjnius (only if your Pydroid build does not already provide it)

Android permissions:
    RECORD_AUDIO
    CAMERA

For a distributable APK, python-for-android/Buildozer is preferable.
"""

import os, json, time, random, math, threading
from collections import deque, Counter
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp, sp
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView

# ---------------- configuration ----------------

N_TOTAL, N_IN, N_OUT, N_HID = 120_000, 1_000, 100, 118_900
N_EAR, N_VIS = 64, 48
FAN_H, FAN_O = 8, 64
FS, CHUNK = 16000, 1024
VREST, VRESET, VTH = -65.0, -68.0, -50.0
TAU, REFRACT = 20.0, 2
TRACE, APLUS, FORGET, WMAX = .94, .0007, .000008, .22
# Android packages are read-only; persist learning data in app-private storage.
def memory_file_path():
    try:
        from kivy.app import App
        app = App.get_running_app()
        if app is not None and app.user_data_dir:
            return os.path.join(app.user_data_dir, "charlie_brain.json")
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "charlie_brain.json")


MAX_SOUND, MAX_VIS, MAX_CONCEPT, MAX_SEQ = 256, 256, 256, 512

# These are vocal exploration primitives, not words or commands.
VOCAL_PRIMITIVES = ("mmm","aaa","ooo","uuu","ba","da","ga","ma","na","pa","ta","ka")


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, float(x)))


def cos_sim(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


# ---------------- persistent memory ----------------

class Memory:
    def __init__(self):
        self.path = memory_file_path()
        self.sound, self.visual, self.concepts, self.sequences = [], [], [], {}
        self.episodes = []
        self.stats = dict(hearing=0, seeing=0, vocal=0, novel=0)
        self.load()

    def load(self):
        try:
            if not os.path.exists(self.path): return
            with open(self.path, encoding="utf-8") as f: d = json.load(f)
            self.sound = d.get("sound", [])[-MAX_SOUND:]
            self.visual = d.get("visual", [])[-MAX_VIS:]
            self.concepts = d.get("concepts", [])[-MAX_CONCEPT:]
            self.sequences = d.get("sequences", {})
            self.episodes = d.get("episodes", [])[-600:]
            self.stats.update(d.get("stats", {}))
        except Exception:
            pass

    def save(self):
        d = {"sound":self.sound[-MAX_SOUND:], "visual":self.visual[-MAX_VIS:],
             "concepts":self.concepts[-MAX_CONCEPT:],
             "sequences":dict(list(self.sequences.items())[-MAX_SEQ:]),
             "episodes":self.episodes[-600:], "stats":self.stats}
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass

    @staticmethod
    def nearest(items, vec, threshold):
        best, score = -1, -1
        for i, x in enumerate(items):
            v = x.get("v")
            if v:
                s = cos_sim(vec, v)
                if s > score: best, score = i, s
        return (best, score) if best >= 0 and score >= threshold else (-1, score)


# ---------------- sparse 120K SNN ----------------

class SNN120K:
    def __init__(self, seed=23):
        r = np.random.default_rng(seed)
        self.r = r
        self.v = np.full(N_TOTAL, VREST, np.float32)
        self.sp = np.zeros(N_TOTAL, np.float32)
        self.ref = np.zeros(N_TOTAL, np.int16)
        self.pre = np.zeros(N_TOTAL, np.float32)
        self.post = np.zeros(N_TOTAL, np.float32)
        self.input = np.zeros(N_IN, np.float32)
        self.da, self.hidden_rate, self.steps = 0., 0., 0.

        hs, he = N_IN, N_IN + N_HID
        hnnz, onnz = N_HID*FAN_H, N_OUT*FAN_O
        self.idx = np.empty(hnnz+onnz, np.int32)
        self.w = np.empty(hnnz+onnz, np.float32)

        k = 0
        for row in range(N_HID):
            post = hs + row
            if row < 2048:
                src = np.concatenate((r.integers(0,N_IN,FAN_H//2),
                                      r.integers(hs,he,FAN_H-FAN_H//2)))
            else:
                src = r.integers(hs,he,FAN_H)
            src[src == post] = hs + ((row+1) % N_HID)
            self.idx[k:k+FAN_H] = src
            self.w[k:k+FAN_H] = r.uniform(.004,.030,FAN_H)
            # Dale sign belongs to presynaptic neuron.
            self.w[k:k+FAN_H] *= np.where(src >= hs+int(.8*N_HID), -1., 1.)
            k += FAN_H

        for row in range(N_OUT):
            src = r.integers(hs,he,FAN_O)
            self.idx[k:k+FAN_O] = src
            self.w[k:k+FAN_O] = r.uniform(.003,.020,FAN_O)
            k += FAN_O

    def stimulate(self, x, gain=.8):
        n = min(N_IN, len(x))
        self.input[:n] += np.asarray(x[:n], np.float32)*gain

    def step(self):
        self.steps += 1
        # Fixed-size sparse accumulation. This is intentionally conservative
        # for Android memory usage.
        hs, he = N_IN, N_IN+N_HID
        # Vectorized hot path: the previous Python row loops would be
        # far too slow on Android.
        h_n = N_HID * FAN_H
        src_h = self.sp[self.idx[:h_n]].reshape(N_HID, FAN_H)
        wh = self.w[:h_n].reshape(N_HID, FAN_H)
        cur_h = np.sum(wh * src_h, axis=1, dtype=np.float32)

        base = h_n
        src_o = self.sp[self.idx[base:]].reshape(N_OUT, FAN_O)
        wo = self.w[base:].reshape(N_OUT, FAN_O)
        cur_o = np.sum(wo * src_o, axis=1, dtype=np.float32)

        active = self.ref <= 0
        I = np.zeros(N_TOTAL, np.float32)
        I[:N_IN] = self.input
        I[hs:he] = cur_h * (1+.2*self.da)
        I[he:] = cur_o
        dv = (VREST-self.v)/TAU + I
        self.v[active] += dv[active]
        self.sp.fill(0)
        fired = active & (self.v >= VTH)
        self.sp[fired] = 1
        self.v[fired] = VRESET
        self.ref[self.ref>0] -= 1
        self.ref[fired] = REFRACT

        self.pre *= TRACE; self.post *= TRACE
        self.pre += self.sp; self.post += self.sp

        # Vectorized coarse STDP.
        # Dale sign is preserved: only the magnitude is changed.
        lr = 1 + self.da
        active_pre = self.sp[self.idx]
        dw = APLUS * lr * active_pre
        mag = np.minimum(WMAX, np.abs(self.w) + dw)
        self.w = np.sign(self.w) * mag
        self.w *= (1 - FORGET)
        self.da *= .985
        self.input *= .2
        self.hidden_rate = float(np.mean(self.sp[hs:he]))

    def reward(self, x=.7):
        self.da = clamp(self.da+x, -1, 1)


# ---------------- microphone ----------------

class Mic:
    def __init__(self, callback):
        self.callback, self.rec, self.running = callback, None, False

    def start(self):
        try:
            from jnius import autoclass
            AR = autoclass("android.media.AudioRecord")
            AS = autoclass("android.media.MediaRecorder$AudioSource")
            AF = autoclass("android.media.AudioFormat")
            ch, enc = AF.CHANNEL_IN_MONO, AF.ENCODING_PCM_16BIT
            size = max(int(AR.getMinBufferSize(FS,ch,enc)), CHUNK*4)
            self.rec = AR(AS.MIC, FS, ch, enc, size)
            self.rec.startRecording(); self.running = True
            threading.Thread(target=self.loop, daemon=True).start()
            return True
        except Exception:
            return False

    def loop(self):
        buf = np.zeros(CHUNK, np.int16)
        while self.running:
            try:
                n = self.rec.read(buf,0,CHUNK)
                if n > 0: self.callback(np.array(buf[:n], np.int16))
            except Exception: time.sleep(.1)

    def stop(self):
        self.running = False
        try:
            if self.rec: self.rec.stop(); self.rec.release()
        except Exception: pass
        self.rec = None


# ---------------- audio analysis ----------------

class AudioFeatures:
    def __init__(self):
        self.win = np.hanning(512).astype(np.float32)
        self.rms = 0.; self.voiced = False
        self.vec = np.zeros(N_EAR,np.float32)

    def process(self, pcm):
        x = np.asarray(pcm,np.float32)/32768.
        self.rms = float(np.sqrt(np.mean(x*x)+1e-10))
        self.voiced = self.rms > .012
        if len(x)<512: x=np.pad(x,(0,512-len(x)))
        else: x=x[-512:]
        s=np.log1p(np.abs(np.fft.rfft(x*self.win))[:N_EAR]*20)
        s/=np.max(s)+1e-6
        s=(s-np.mean(s)).astype(np.float32)
        s/=np.linalg.norm(s)+1e-6
        self.vec=s
        return s


# ---------------- vision ----------------

class VisionFeatures:
    def __init__(self):
        self.vec=np.zeros(N_VIS,np.float32); self.motion=0.; self.brightness=0.

    def process(self, texture):
        if texture is None: return self.vec,0.
        try:
            p=np.frombuffer(texture.pixels,np.uint8)
            if len(p)<4: return self.vec,0.
            rgba=p.reshape((-1,4)).astype(np.float32)/255.
            n=min(256,len(rgba))
            sample=rgba[np.linspace(0,len(rgba)-1,n).astype(np.int32),:3]
            gray=sample.mean(1)
            chunks=np.array_split(gray,32)
            f=list(sample.mean(0))+list(sample.std(0))+[float(x.mean()) for x in chunks]
            v=np.asarray(f[:N_VIS],np.float32)
            if len(v)<N_VIS: v=np.pad(v,(0,N_VIS-len(v)))
            v-=v.mean(); v/=np.linalg.norm(v)+1e-6
            self.motion=float(np.linalg.norm(v-self.vec))
            self.brightness=float(sample.mean())
            self.vec=v
        except Exception: pass
        return self.vec,self.motion


# ---------------- developmental brain ----------------

class Brain:
    def __init__(self):
        self.mem=Memory(); self.snn=SNN120K()
        self.audio=AudioFeatures(); self.visual=VisionFeatures()
        self.lock=threading.RLock()
        self.sound_id=self.visual_id=None
        self.sound_seq=deque(maxlen=12)
        self.novelty=self.curiosity=self.uncertainty=self.social=self.vocal=0.
        self.last_event="waiting"; self.last_state="quiet"
        self.last_speech=0.; self.last_initiative=0.
        self.vocal_history=deque(maxlen=20)
        self.stage="сенсорное исследование"

    def learn_sound(self,v,voiced):
        if not voiced:return
        i,sim=Memory.nearest(self.mem.sound,v,.86)
        if i<0:
            if len(self.mem.sound)>=MAX_SOUND:
                i=min(range(len(self.mem.sound)),key=lambda j:self.mem.sound[j]["n"])
                self.mem.sound[i]={"v":v.tolist(),"n":1}
            else:
                self.mem.sound.append({"v":v.tolist(),"n":1}); i=len(self.mem.sound)-1
            self.novelty=1.; self.mem.stats["novel"]+=1
        else:
            q=np.asarray(self.mem.sound[i]["v"],np.float32)
            q=.97*q+.03*v; q/=np.linalg.norm(q)+1e-6
            self.mem.sound[i]["v"]=q.tolist(); self.mem.sound[i]["n"]+=1
            self.novelty*=.94
        self.sound_id=i; self.sound_seq.append(i)
        self.mem.stats["hearing"]+=1
        self.learn_sequences()
        self.uncertainty=clamp(1-(sim if sim>=0 else 0))

    def learn_sequences(self):
        a=list(self.sound_seq)
        for n in (2,3,4):
            if len(a)<n:continue
            k="-".join(map(str,a[-n:]))
            x=self.mem.sequences.setdefault(k,{"n":0,"len":n})
            x["n"]+=1

    def learn_visual(self,v,motion):
        i,sim=Memory.nearest(self.mem.visual,v,.92)
        if i<0:
            if len(self.mem.visual)>=MAX_VIS:
                i=min(range(len(self.mem.visual)),key=lambda j:self.mem.visual[j]["n"])
                self.mem.visual[i]={"v":v.tolist(),"n":1}
            else:
                self.mem.visual.append({"v":v.tolist(),"n":1}); i=len(self.mem.visual)-1
            self.novelty=max(self.novelty,.7)
        else:
            q=np.asarray(self.mem.visual[i]["v"],np.float32)
            q=.98*q+.02*v; q/=np.linalg.norm(q)+1e-6
            self.mem.visual[i]["v"]=q.tolist(); self.mem.visual[i]["n"]+=1
        self.visual_id=i; self.mem.stats["seeing"]+=1
        self.curiosity=clamp(self.curiosity+.03*motion)

    def associate(self):
        if self.sound_id is None or self.visual_id is None:return
        a,b=self.sound_id,self.visual_id
        x=next((c for c in self.mem.concepts if c["s"]==a and c["v"]==b),None)
        if x is None:
            self.mem.concepts.append({"s":a,"v":b,"n":1,"c":.08})
        else:
            x["n"]+=1; x["c"]=clamp(1-math.exp(-x["n"]/12))

    def drives(self,human):
        self.novelty*=.99; self.uncertainty*=.995; self.curiosity*=.995
        self.social=clamp(self.social+.012) if human else self.social*.995
        self.vocal=clamp(.35*self.curiosity+.3*self.uncertainty+
                          .2*self.social+.15*self.novelty)
        self.last_state=("потребность в общении" if self.vocal>.7 else
                         "curious" if self.curiosity>.55 else
                         "uncertain" if self.uncertainty>.55 else "quiet")
        units=len(self.mem.sound); concepts=len(self.mem.concepts)
        seq=sum(1 for x in self.mem.sequences.values() if x["n"]>=3)
        if units<5:self.stage="сенсорное исследование"
        elif units<20:self.stage="вокальная и перцептивная игра"
        elif concepts<5:self.stage="обучение ассоциациям"
        elif seq<5:self.stage="поиск последовательностей"
        elif seq<30:self.stage="первые комбинации"
        else:self.stage="формирование речи"

    def vocalize(self,force=False):
        if not force:
            if time.time()-self.last_initiative<12:return None
            if time.time()-self.last_speech<8:return None
            if random.random() > .018+.11*self.vocal:return None
        if self.vocal_history and random.random()<.55:
            s=Counter(self.vocal_history).most_common(1)[0][0]
        else:s=random.choice(VOCAL_PRIMITIVES)
        self.vocal_history.append(s); self.mem.stats["vocal"]+=1
        self.last_speech=time.time(); self.last_initiative=time.time()
        return s

    def episode(self,human):
        self.mem.episodes.append({"t":time.time(),"s":self.sound_id,"v":self.visual_id,
                                  "human":bool(human),"novel":round(self.novelty,3),
                                  "unc":round(self.uncertainty,3)})
        self.mem.episodes=self.mem.episodes[-600:]


# ---------------- Android voice ----------------

class Voice:
    def __init__(self):self.tts=None
    def init(self):
        try:
            from jnius import autoclass
            P=autoclass("org.kivy.android.PythonActivity")
            T=autoclass("android.speech.tts.TextToSpeech")
            L=autoclass("java.util.Locale")
            self.tts=T(P.mActivity,None); self.tts.setLanguage(L("ru","RU")); return True
        except Exception:return False
    def speak(self,x):
        try:
            if self.tts is None and not self.init():return False
            T=__import__("jnius",fromlist=["autoclass"]).autoclass("android.speech.tts.TextToSpeech")
            self.tts.speak(str(x),T.QUEUE_FLUSH,None); return True
        except Exception:return False
    def close(self):
        try:self.tts.stop();self.tts.shutdown()
        except Exception:pass


# ---------------- UI ----------------

class Meter(BoxLayout):
    def __init__(self,name,label=None):
        super().__init__(orientation="horizontal",size_hint_y=None,height=dp(28),spacing=dp(5))
        self.add_widget(Label(text=(label or name),size_hint_x=.38,font_size=sp(11)))
        self.bar=ProgressBar(max=1,value=0,size_hint_x=.62);self.add_widget(self.bar)
    def set(self,x):self.bar.value=clamp(x)


class CharlieUI(BoxLayout):
    status=StringProperty("запуск")

    def __init__(self,**kw):
        super().__init__(orientation="vertical",padding=dp(7),spacing=dp(6),**kw)
        # IMPORTANT FOR ANDROID:
        # Do not allocate the 120k-neuron brain or open Camera before the
        # first Kivy frame. The old version could remain on a black screen
        # and be killed by Android before drawing anything.
        self.brain=None
        self.voice=Voice()
        self.mic=None
        self.cam=None
        self.running=False
        self.last_vis=0
        self.human=False
        self.brain_ready=False
        self.build_ui()
        Clock.schedule_once(self.begin_brain_init, .15)
        Clock.schedule_interval(self.tick,.12)
        Clock.schedule_interval(self.refresh,.3)

    def build_ui(self):
        self.add_widget(Label(text="[b]ЧАРЛИ[/b]  •  развивающийся мультимодальный мозг",
                              markup=True,font_size=sp(20),size_hint_y=None,height=dp(34)))
        self.state=Label(text="запуск интерфейса…",font_size=sp(12),size_hint_y=None,height=dp(28))
        self.add_widget(self.state)

        # Safe placeholder. Camera is created only after START.
        self.camera_box=BoxLayout(size_hint_y=.38)
        self.camera_placeholder=Label(text="КАМЕРА\nзапустится после нажатия «СТАРТ»",halign="center",valign="middle")
        self.camera_placeholder.bind(size=lambda *_: setattr(
            self.camera_placeholder,"text_size",self.camera_placeholder.size))
        self.camera_box.add_widget(self.camera_placeholder)
        self.add_widget(self.camera_box)

        g=GridLayout(cols=2,size_hint_y=None,height=dp(140),spacing=dp(3))
        self.meters={}
        meter_names=(
            ("novelty","Новизна"),("uncertainty","Неопределённость"),
            ("curiosity","Любопытство"),("social","Социальность"),
            ("vocal drive","Потребность говорить"),("microphone","Микрофон")
        )
        for key,label in meter_names:
            self.meters[key]=Meter(key,label);g.add_widget(self.meters[key])
        self.add_widget(g)

        row=GridLayout(cols=4,size_hint_y=None,height=dp(46),spacing=dp(5))
        self.start=Button(text="СТАРТ",disabled=True)
        self.start.bind(on_release=lambda *_:self.toggle())
        save=Button(text="СОХРАНИТЬ",disabled=True)
        save.bind(on_release=lambda *_:self.save())
        reward=Button(text="НАГРАДА",disabled=True)
        reward.bind(on_release=lambda *_:self.reward())
        self.save_button=save; self.reward_button=reward
        access=Button(text="ДОСТУП")
        access.bind(on_release=lambda *_:self.open_android_settings())
        row.add_widget(self.start);row.add_widget(save);row.add_widget(reward);row.add_widget(access)
        self.add_widget(row)

        sv=ScrollView(size_hint_y=.30)
        self.info=Label(text="Подготовка Чарли…",font_size=sp(11),halign="left",valign="top",text_size=(None,None))
        sv.add_widget(self.info);self.add_widget(sv)

    def begin_brain_init(self,dt):
        self.status="загрузка мозга…"
        def worker():
            try:
                b=Brain()
                Clock.schedule_once(lambda _dt:self.brain_initialized(b),0)
            except Exception as e:
                msg="ошибка запуска мозга: "+type(e).__name__
                Clock.schedule_once(lambda _dt:self.brain_failed(msg),0)
        threading.Thread(target=worker,daemon=True).start()

    def brain_initialized(self,b):
        self.brain=b
        self.brain_ready=True
        self.status="готово"
        self.state.text="готово — нажмите «СТАРТ»"
        self.start.disabled=False
        self.save_button.disabled=False
        self.reward_button.disabled=False
        self.info.text="Чарли готова. Нажмите «СТАРТ», чтобы начать обучение."

    def brain_failed(self,msg):
        self.status=msg
        self.state.text=msg
        self.info.text=("Не удалось запустить нейронное ядро.\n"
                        "Интерфейс продолжает работать, поэтому приложение не закрывается.\n\n"+msg)

    # ---------------- Android permissions ----------------

    def _permission_state(self):
        """Return current Android runtime permission state when available."""
        try:
            from android.permissions import check_permission, Permission
            return (bool(check_permission(Permission.RECORD_AUDIO)),
                    bool(check_permission(Permission.CAMERA)))
        except Exception:
            # Some Pydroid builds do not expose check_permission.
            # In that case we let the actual sensor open decide.
            return (None, None)

    def open_android_settings(self):
        """Open the Android application settings for Pydroid/current app."""
        try:
            from jnius import autoclass
            PythonActivity=autoclass("org.kivy.android.PythonActivity")
            Intent=autoclass("android.content.Intent")
            Settings=autoclass("android.provider.Settings")
            activity=PythonActivity.mActivity
            intent=Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
            intent.setData(autoclass("android.net.Uri").parse(
                "package:" + str(activity.getPackageName())))
            activity.startActivity(intent)
            self.info.text=("Откройте «Разрешения» и включите КАМЕРУ и МИКРОФОН.\n"
                            "После возврата нажмите «СТАРТ» ещё раз.")
        except Exception as e:
            self.info.text=("Не удалось открыть настройки автоматически.\n"
                            "Откройте Android → Настройки → Приложения → Pydroid 3 → Разрешения\n"
                            "и включите КАМЕРУ и МИКРОФОН.\n\n" + type(e).__name__)

    def request_sensor_permissions(self, after=None):
        """Request permissions and continue only after Android answers."""
        try:
            from android.permissions import request_permissions, Permission
            audio_ok, camera_ok = self._permission_state()

            needed=[]
            if audio_ok is not True: needed.append(Permission.RECORD_AUDIO)
            if camera_ok is not True: needed.append(Permission.CAMERA)

            if not needed:
                if after: Clock.schedule_once(lambda _dt: after(True, True), 0)
                return

            self.status="запрос доступа к камере и микрофону…"
            self.state.text="Разрешите Чарли доступ к камере и микрофону"

            def result_callback(permissions, grants):
                # grants may be Java/Python booleans depending on Android build.
                audio_granted=False
                camera_granted=False
                for perm, grant in zip(permissions, grants):
                    name=str(perm)
                    ok=bool(grant)
                    if "RECORD_AUDIO" in name: audio_granted=ok
                    if "CAMERA" in name: camera_granted=ok

                # Re-check the real state because some Android versions return
                # integer constants rather than Python bools in the callback.
                a2,c2=self._permission_state()
                if a2 is not None: audio_granted=audio_granted or a2
                if c2 is not None: camera_granted=camera_granted or c2
                Clock.schedule_once(lambda _dt: self._permissions_finished(
                    audio_granted,camera_granted,after),0)

            request_permissions(needed,result_callback)
        except Exception as e:
            self._permissions_error=type(e).__name__
            Clock.schedule_once(lambda _dt: self._permissions_finished(
                False,False,after),0)

    def _permissions_finished(self,audio_ok,camera_ok,after=None):
        if audio_ok and camera_ok:
            self.state.text="Доступ к камере и микрофону разрешён"
        elif audio_ok:
            self.state.text="Микрофон разрешён • камера запрещена или недоступна"
        elif camera_ok:
            self.state.text="Камера разрешена • микрофон запрещён или недоступен"
        else:
            self.state.text=("Доступ к камере и микрофону не получен.\n"
                             "Разрешите их в настройках Android для Pydroid 3.")
        if after:
            try: after(audio_ok,camera_ok)
            except Exception: pass

    def ensure_camera(self):
        if self.cam is not None:return True
        try:
            from kivy.uix.camera import Camera
            cam=Camera(play=False,resolution=(320,240),size_hint=(1,1))
            self.camera_box.clear_widgets()
            self.camera_box.add_widget(cam)
            self.cam=cam
            # Camera provider can need a frame or two before texture appears.
            Clock.schedule_once(lambda _dt: self._camera_status_check(),1.2)
            return True
        except Exception as e:
            self.cam=None
            self.camera_box.clear_widgets()
            self.camera_placeholder.text=("КАМЕРА НЕДОСТУПНА\n"
                                          "Проверьте разрешение CAMERA в Android")
            self.camera_box.add_widget(self.camera_placeholder)
            if self.brain:self.brain.last_event="камера недоступна"
            return False

    def _camera_status_check(self):
        if not self.running or self.cam is None:return
        try:
            tex=self.cam.texture
            if tex is None:
                self.camera_placeholder.text=("КАМЕРА НЕ ПОЛУЧАЕТ ИЗОБРАЖЕНИЕ\n"
                                              "Проверьте доступ к камере в Android")
        except Exception: pass

    def toggle(self):
        if not self.brain_ready:return
        if self.running:self.stop()
        else:self.start_learning()

    def start_learning(self):
        if not self.brain:return
        # Do NOT start sensors until Android has completed the permission dialog.
        self.start.disabled=True
        self.status="проверка разрешений…"
        self.request_sensor_permissions(self._start_sensors)

    def _start_sensors(self,audio_permission,camera_permission):
        if not self.brain:return
        self.running=True
        self.start.text="СТОП"
        self.start.disabled=False
        self.status="запуск обучения…"

        camera_ok=False
        mic_ok=False

        if camera_permission:
            camera_ok=self.ensure_camera()
            if camera_ok:
                try:self.cam.play=True
                except Exception: camera_ok=False
        else:
            self.camera_box.clear_widgets()
            self.camera_placeholder.text=("КАМЕРА ОТКЛЮЧЕНА\n"
                                          "Разрешите CAMERA в настройках Android")
            self.camera_box.add_widget(self.camera_placeholder)

        if audio_permission:
            try:
                self.mic=Mic(self.audio_callback)
                mic_ok=self.mic.start()
            except Exception:
                self.mic=None; mic_ok=False
        else:
            self.mic=None

        self.voice.init()
        if mic_ok and camera_ok:
            self.status="обучение • камера + микрофон"
        elif mic_ok:
            self.status="обучение • микрофон работает, камера недоступна"
        elif camera_ok:
            self.status="обучение • камера работает, микрофон недоступен"
        else:
            self.status="обучение • сенсоры недоступны"

        self.info.text=(
            "Сенсоры: микрофон — %s; камера — %s.\n"
            "Чарли продолжает обучение доступными каналами." %
            ("РАБОТАЕТ" if mic_ok else "НЕДОСТУПЕН",
             "РАБОТАЕТ" if camera_ok else "НЕДОСТУПНА"))

    def stop(self):
        self.running=False
        self.start.text="СТАРТ"
        try:
            if self.cam:self.cam.play=False
        except Exception:pass
        if self.mic:
            try:self.mic.stop()
            except Exception:pass
        self.mic=None
        if self.brain:
            try:self.brain.mem.save()
            except Exception:pass
        self.status="пауза"

    def audio_callback(self,pcm):
        b=self.brain
        if not self.running or b is None:return
        try:
            v=b.audio.process(pcm)
            b.learn_sound(v,b.audio.voiced)
            x=np.zeros(N_IN,np.float32)
            for i in range(N_EAR):
                a=i*N_IN//N_EAR;bnd=(i+1)*N_IN//N_EAR
                x[a:bnd]=abs(float(v[i]))*.06
            b.snn.stimulate(x)
        except Exception:
            pass

    def tick(self,dt):
        b=self.brain
        if not self.running or b is None:return
        try:
            b.snn.step()
            if self.cam is not None and time.time()-self.last_vis>.45:
                self.last_vis=time.time()
                v,m=b.visual.process(self.cam.texture)
                b.learn_visual(v,m)
                b.associate()
            b.human=self.human
            self.human=(b.visual.brightness>.055)
            b.drives(self.human)
            b.episode(self.human)
            if self.human:
                s=b.vocalize()
                if s:
                    self.voice.speak(s)
                    b.last_event="самостоятельная вокализация: "+s
        except Exception:
            # Never let one sensor/SNN frame kill the Android UI.
            b.last_event="кадр пропущен"

    def refresh(self,dt):
        b=self.brain
        if b is None:
            self.state.text=self.status
            return
        for k,v in (("novelty",b.novelty),("uncertainty",b.uncertainty),
                    ("curiosity",b.curiosity),("social",b.social),
                    ("vocal drive",b.vocal),("microphone",b.audio.rms*25)):
            self.meters[k].set(v)
        self.state.text=f"{self.status}   •   этап: {b.stage}"
        seq=sum(1 for x in b.mem.sequences.values() if x["n"]>=3)
        self.info.text=(
            "[b]ВНУТРЕННЕЕ СОСТОЯНИЕ[/b]\n"
            f"состояние: {b.last_state}\n"
            f"событие: {b.last_event}\n"
            f"этап: {b.stage}\n\n"
            "[b]НАКОПЛЕННЫЙ ОПЫТ[/b]\n"
            f"звуковые единицы: {len(b.mem.sound)}\n"
            f"визуальные единицы: {len(b.mem.visual)}\n"
            f"мультимодальные понятия: {len(b.mem.concepts)}\n"
            f"повторяющиеся последовательности: {seq}\n"
            f"события слуха: {b.mem.stats['hearing']}\n"
            f"события зрения: {b.mem.stats['seeing']}\n"
            f"самостоятельные вокализации: {b.mem.stats['vocal']}\n"
            f"новые события: {b.mem.stats['novel']}\n\n"
            "[b]НЕЙРОННАЯ СЕТЬ[/b]\n"
            f"нейронов: {N_TOTAL:,}\n"
            f"активность скрытых нейронов: {b.snn.hidden_rate:.5f}\n"
            f"дофамин: {b.snn.da:.3f}\n\n"
            "Встроенных русских слов и вопросов нет.\n"
            "Изначально доступна только примитивная вокальная разведка."
        )

    def save(self):
        if self.brain:
            self.brain.mem.save();self.status="память сохранена"

    def reward(self):
        if self.brain:
            self.brain.snn.reward(.7);self.brain.social=clamp(self.brain.social+.15)
            self.status="получена награда"

    def close(self):
        try:self.stop()
        except Exception:pass
        try:self.voice.close()
        except Exception:pass



class CharlieApp(App):
    title="Чарли — развивающийся мозг"
    def build(self):
        self.ui=CharlieUI();return self.ui
    def on_stop(self):
        self.ui.close()


if __name__=="__main__":
    CharlieApp().run()
