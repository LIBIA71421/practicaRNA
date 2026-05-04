# Practica RNA - MLPs con FIFA 2021

Implementacion de la guia `practicaRNA.pdf` usando PyTorch y el dataset `fifa2021_training.csv`.

## Que incluye

- Carga y preprocesamiento del dataset (columnas base + habilidades, one-hot en `Sex`).
- Split 70/30 con `train_test_split` y estandarizacion de features.
- DataLoaders con `batch_size=32`.
- Modelos base solicitados:
  - Chico: `[4, 4]`
  - Medio: `[16, 16]`
  - Grande: `[256, 256]`
- Entrenamiento con `CrossEntropyLoss` + `Adam`.
- Registro en TensorBoard de loss y accuracy (train/test) con tags jerarquicos y frecuencia configurable.
- Bloque opcional de "competencia" con 5 arquitecturas configurables.

## Instalacion

```powershell
pip install -r requirements.txt
```

## Ejecucion

Ejecucion rapida (prueba):

```powershell
python main.py --base-epochs 5
```

Ejecucion completa de modelos base (como la guia):

```powershell
python main.py --base-epochs 500
```

Ejecucion con menos ruido visual en TensorBoard (log cada 10 epocas):

```powershell
python main.py --base-epochs 500 --tb-log-every 10
```

Ejecucion con competencia (5 arquitecturas):

```powershell
python main.py --base-epochs 500 --run-competition --competition-epochs 300
```

## TensorBoard

```powershell
tensorboard --logdir runs
cmd /c "C:\Users\admin\PycharmProjects\practicaRNA\open_tensorboard.bat"

Set-Location "C:\Users\admin\PycharmProjects\practicaRNA"; python -m tensorboard.main --logdir "runs\20260504-192121" --host localhost --port 6007
```

Tambien puedes abrir TensorBoard con doble clic en Windows:

- `open_tensorboard.bat`: opcion recomendada si quieres evitar problemas de politica de ejecucion de PowerShell.
- `open_tensorboard.ps1`: opcion directa en PowerShell.

Cada ejecucion crea una subcarpeta con timestamp dentro de `runs/`, y dentro de ella se guardan los logs de `base/chico`, `base/medio`, `base/grande` y, si aplica, `competition/...`.

Para evitar graficos saturados, abre TensorBoard apuntando a una corrida especifica (la ruta exacta tambien se imprime por consola):

```powershell
tensorboard --logdir "runs\YYYYMMDD-HHMMSS"
```

El script tambien imprime en consola la ruta exacta donde va registrando cada modelo.

Los scripts abren TensorBoard en segundo plano y luego intentan abrir automaticamente el navegador en `http://localhost:6006` o en el siguiente puerto libre.

Luego abrir en navegador la URL que imprime TensorBoard (normalmente `http://localhost:6006`).

## Validacion de estilo (PEP 8)

```powershell
python -m pycodestyle main.py
```

## Notas

- El script usa `cuda` automaticamente si esta disponible, o `cpu` en caso contrario.
- Si quieres forzar CPU:

```powershell
python main.py --device cpu
```

