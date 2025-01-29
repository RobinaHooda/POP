# POP

*Autorzy*
- *Jakub Bąba*
- *Aleksandra Szymańska*

Repozytorium zawiera projekt zaliczeniowy z przedmiotu **Przeszukiwanie i optymalizacja (POP)**. W folderze docs znajduje się dokumentacja wstępna.

## Instrukcja uruchomienia
Wymaganie wstępne: `python` musi być zainstalowany.

Aby uruchomić program, należy wykonać poniższe komendy:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiment.py
python analysis.py
```

Alternatywnie, program można uruchomić za pomocą jednej komendy:
```bash
bash run.sh
```
Skrypt w zautomatyzowany sposób sprawdza czy utworzone jest środowisko wirtualne, aktualizuje zależności i uruchamia programy.
