## TP3: Perceptrón Simple y Multicapa

## Autores: 
- Juan Dusau 
- Nicolás Urbano Pintos

## Requerimientos

Primero es necesario clonar el respositorio:

```console
git clone https://github.com/nurbano/sia
```
Luego hay que ir al directorio tp3:

```console
cd sia/tp3
```
Es necesario instalar primero los paquetes requeridos:

```console
pip install -r requirements.txt
```

## Ejercicio 1
```console
python ej1.py --config_json ./config/config_and.json
python ej1.py --config_json ./config/config_xor.json
```

## Ejercicio 2
```console
python ej2.py --config_json ./config/config_ej2_norm_lineal.json
python ej2.py --config_json ./config/config_ej2_norm_no_lineal.json
python ej2.py --config_json ./config/config_ej2_sin_norm_lineal.json
python ej2.py --config_json ./config/config_ej2_sin_norm_no_lineal.json
```

## Ejercicio 3
```console
python perc_multi_paridad.py --config_json ./config/config_digitos_paridad.json
python perc_multi_paridad_ruido.py --config_json ./config/config_digitos_con_ruido_paridad.json
python perc_multi_digitos.py --config_json ./config/config_digitos_clasificacion.json
```

## Ejercicio 4
```console
python ej4.py
```