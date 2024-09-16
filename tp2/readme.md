## TP2: Algortimos genéticos

## Autores: 
- Juan Dusau 
- Nicolás Urbano Pintos

## Requerimientos

Primero es necesario clonar el respositorio:

```console
git clone https://github.com/nurbano/sia
```
Luego hay que ir al directorio tp2:

```console
cd sia/tp2
```
Es necesario instalar primero los paquetes requeridos:

```console
pip install -r requirements.txt
```

## Uso de ITBUM

Para ejecturar el programa utilizo:

```console
python .\main.py --help                        
usage: main.py [-h] [--config_json CONFIG_JSON]

TP2_AG Autores: - Juan Dusau - Nicolás Urbano Pintos

options:
  -h, --help            show this help message and exit
  --config_json CONFIG_JSON
                        Path to json config file
```
Por ejemplo: 
```console
python .\main.py --config_json .\config\mutacion_3.json
```

## Archivo de Configuración
Es necesario generar un config file, con los hiperparámetros, en el directorio config hay diferentes ejemplos. 
A continuación se detalla un ejemplo

```json
{
    "total_puntos": 100, 
    "poblacion": 100,
    "tiempo": 600,
    "max_generaciones": 50,
    "clase": "mago",
    "delta_fitness_min": 0.001,
    "max_estancamiento": 10,
    "porcentaje_hijos": 0.5,
    "prop_sel_padres": 0.7,
    "T0" : 100,  
    "Tc" : 1,  
    "t" : 10,
    "tam_torneo": 3,
    "th_torneo": 0.3,
    "prop_reemplazo": 0.5,
    "metodo_seleccion_1": "ruleta",
    "metodo_seleccion_2": "torneo_deterministico",
    "prob_mutacion": 0.1
}
```