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
- Registro en TensorBoard de loss y accuracy (train/test).
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

Ejecucion con competencia (5 arquitecturas):

```powershell
python main.py --base-epochs 500 --run-competition --competition-epochs 300
```

## TensorBoard

```powershell
tensorboard --logdir runs
```

Luego abrir en navegador la URL que imprime TensorBoard (normalmente `http://localhost:6006`).

## Notas

- El script usa `cuda` automaticamente si esta disponible, o `cpu` en caso contrario.
- Si quieres forzar CPU:

```powershell
python main.py --device cpu
```

