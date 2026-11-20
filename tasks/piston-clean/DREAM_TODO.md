# Do zrobienia przed zgłoszeniem: przebieg Dream-1 (minimum konkursowe)

Plik roboczy zespołu — **usunąć przed otwarciem PR** (nie jest częścią zadania).

Wymóg z README benchmarku: wideo z przebiegu modelu (`--video`) jest obowiązkowe, demo (`--demo`) jest wymagane,
gdy Dream-1 nie osiąga pełnego wyniku, a pole `scores` w `task.json` podaje oba wyniki tak, jak zgłosił je scorer.

## 1. Klucz Dream-1 (raz)

Klucz z https://platform.dreamlabs.ai/api-keys, wpisany ręcznie na maszynie, na której liczy się symulacja
(na GB10: konto `ljaniec`). Nigdy w repo ani w czacie.

```bash
mkdir -p ~/.dreamlabs && echo 'DREAM_API_KEY=dl_live_...' > ~/.dreamlabs/credentials && chmod 600 ~/.dreamlabs/credentials
```

## 2. Przebieg Dream-1 (ok. 15 min na GB10, koszt ok. 0,20 USD: 4800 kroków = 4 min sterowania po 3 USD/h)

```bash
source ~/robocurve/IsaacLab-Arena/yam.env
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"   # tylko na ARM (GB10)
cd ~/robocurve/yam-isaac-lab-arena
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$HOME/robocurve/tasks/piston-clean/piston_clean.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Clean the oil off the piston." \
  --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/piston-dream
```

Na końcu przebieg wypisuje `task_score: X` — to liczba do `scores`.

## 3. Filmy (H.264, do 50 MB każdy)

```bash
for log in ~/yam-eval/piston-dream/*.json; do inspect-robots video "$log"; done
ffmpeg -i <katalog frames>/*top_cam.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart model_run.mp4
```

Demo (`demo.mp4`, wynik 1.0) jest już nagrane: `D:\robocurve\media\demo-submission.mp4` na Shadow,
`~/robocurve/demo.mp4` na GB10. Zrzut układu startowego zrobić z pierwszej klatki (`setup.png`), najlepiej w 640 px.

## 4. task.json

```json
"scores": "Dream-1 run: X; demo: 1.0"
```

## 5. Wgranie mediów i PR (zamraża zadanie do recenzji — tylko za zgodą zespołu)

```bash
export ROBOCURVE_TEAM=T-.. ROBOCURVE_TOKEN=...     # kod i token zespołu, nie commitować
python tools/upload_media.py tasks/piston-clean --video model_run.mp4 --demo demo.mp4 --screenshot setup.png
```

Potem fork `robocurve/benchmark-tasks` z konta zarejestrowanego przez zespół, gałąź z folderem
`tasks/piston-clean` (bez tego pliku), `gh pr create --repo robocurve/benchmark-tasks --fill`.
Przed PR: ślepa recenzja — w `task.json`, zrzucie i filmach nic, co identyfikuje zespół.
