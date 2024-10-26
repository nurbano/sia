## TP4: Aprendizaje no supervisado

## Autores: 
- Juan Dusau 
- Nicolás Urbano Pintos

## Requerimientos

Primero es necesario clonar el respositorio:

```console
git clone https://github.com/nurbano/sia
```
Luego hay que ir al directorio tp4:

```console
cd sia/tp4
```
Es necesario instalar primero los paquetes requeridos:

```console
pip install -r requirements.txt
```
## OJA

```console
python oja.py --config_json .\config\config_oja.json    
```

## Kohonen
```console
python kohonen.py --config_json .\config\config_kohonen.json
```

## Hopfield
```console
python hopfield.py --config_json .\config\config_hopfield_best.json
```