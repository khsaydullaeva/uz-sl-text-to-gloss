# Inference

```bash
# from this directory
PYTHONPATH=../src python main.py --text "Men uyda o'tiribman." --model nllb
PYTHONPATH=../src python main.py --text "..." --model all --json
PYTHONPATH=../src python ui.py --open
```

Checkpoints are resolved from `../artifacts/ckpts/<model>/best/`.
