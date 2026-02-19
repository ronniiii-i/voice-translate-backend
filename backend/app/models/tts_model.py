import subprocess
import wave
import threading
import time
import select
from pathlib import Path


class PiperTTS:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent.parent
        self.model_dir = self.root_dir / "models/tts"

        self.model_map = {
            "en": self._pick("en_US-ryan-low.onnx",     "en_US-bryce-medium.onnx"),
            "fr": self._pick("fr_FR-siwis-low.onnx",    "fr_FR-siwis-medium.onnx"),
            "de": self._pick("de_DE-thorsten-low.onnx",  "de_DE-thorsten-medium.onnx"),
            "es": self._pick("es_ES-mls_10246-low.onnx", "es_ES-mls_10246-low.onnx"),
        }

        self._procs: dict[str, subprocess.Popen] = {}
        self._locks: dict[str, threading.Lock] = {
            lang: threading.Lock() for lang in self.model_map
        }

        for lang, path in self.model_map.items():
            quality = "low ✅" if "low" in str(path) else "medium ⚠️ (slow)"
            status  = "found" if path.exists() else "MISSING ❌"
            print(f"[TTS] {lang}: {path.name} [{quality}] [{status}]")

    def _pick(self, *candidates: str) -> Path:
        for name in candidates:
            p = self.model_dir / name
            if p.exists():
                return p
        return self.model_dir / candidates[-1]

    def _model_path(self, language: str) -> Path:
        path = self.model_map.get(language, self.model_map["en"])
        if not path.exists():
            en = self.model_map["en"]
            if en.exists():
                print(f"[TTS] ⚠️  No model for '{language}', using English")
                return en
            raise FileNotFoundError(
                f"No TTS model for '{language}' and English fallback also missing."
            )
        return path

    def _get_proc(self, language: str) -> subprocess.Popen:
        """Return the live persistent process, (re)starting it if dead."""
        proc = self._procs.get(language)
        if proc is not None and proc.poll() is None:
            return proc

        cmd = [
            "piper",
            "--model",            str(self._model_path(language)),
            "--output-raw",       
            "--sentence-silence", "0.1",
        ]

        print(f"[TTS] Starting Piper [{language}]...")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.4)   # let Piper load the model
        if proc.poll() is not None:
            err = proc.stderr.read().decode(errors="replace")[:300]
            raise RuntimeError(f"Piper [{language}] failed to start: {err}")

        self._procs[language] = proc
        print(f"[TTS] ✅ Piper [{language}] ready (pid={proc.pid})")
        return proc

    def _read_utterance(self, proc: subprocess.Popen, timeout: float = 10.0) -> bytes:
        """
        Read one complete utterance from Piper stdout.
        We collect PCM chunks until data stops flowing for ~100ms.
        """
        chunks = []
        deadline   = time.time() + timeout
        idle_wait  = 0.20   # initial: wait up to 200ms for first byte

        while time.time() < deadline:
            ready, _, _ = select.select([proc.stdout], [], [], idle_wait)
            if ready:
                chunk = proc.stdout.read1(16384)
                if chunk:
                    chunks.append(chunk)
                    idle_wait = 0.08    # tighten window once audio starts flowing
                else:
                    break   # pipe closed
            else:
                if chunks:
                    break   # no new data for idle_wait seconds → utterance done
                if proc.poll() is not None:
                    raise RuntimeError("Piper process died before producing audio")

        return b"".join(chunks)

    def _kill_proc(self, language: str):
        proc = self._procs.pop(language, None)
        if proc:
            try:
                proc.kill()
            except Exception:
                pass

    def synthesize(self, text: str, output_path: str, language: str = "en") -> str:
        if len(text) > 200:
            trunc = text[:200]
            for punct in ('.', '!', '?', ','):
                idx = trunc.rfind(punct)
                if idx > 80:
                    trunc = trunc[:idx + 1]
                    break
            print(f"[TTS] ✂️  Truncated {len(text)} → {len(trunc)} chars")
            text = trunc

        text = text.replace("\n", " ").replace("\r", " ").strip()
        if not text:
            raise ValueError("[TTS] Empty text after cleaning")

        lock = self._locks.get(language) or threading.Lock()
        t0 = time.time()

        with lock:
            for attempt in range(2):
                try:
                    proc = self._get_proc(language)
                    proc.stdin.write((text + "\n").encode("utf-8"))
                    proc.stdin.flush()
                    raw_pcm = self._read_utterance(proc, timeout=10.0)
                    break
                except Exception as e:
                    print(f"[TTS] ❌ attempt {attempt+1} failed: {e}")
                    self._kill_proc(language)
                    if attempt == 1:
                        raise RuntimeError(f"[TTS] Piper [{language}] failed twice: {e}")

        if not raw_pcm:
            raise RuntimeError(f"[TTS] No PCM output for: '{text[:40]}'")

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)

        elapsed = time.time() - t0
        flag = "✅" if elapsed < 4 else "⚠️ "
        print(f"[TTS] {flag} {elapsed:.1f}s → {Path(output_path).name}")
        return output_path

    def shutdown(self):
        """Call this on app shutdown to cleanly close Piper processes."""
        for lang, proc in list(self._procs.items()):
            try:
                proc.stdin.close()
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        self._procs.clear()
        print("[TTS] All Piper processes stopped.")