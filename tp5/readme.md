## TP5: Deep Learning

## Autores: 
- Juan Dusau 
- Nicolás Urbano Pintos

## Requerimientos

Primero es necesario clonar el respositorio:

```console
git clone https://github.com/nurbano/sia
```
Luego hay que ir al directorio tp5:

```console
cd sia/tp5
```
Es necesario instalar primero los paquetes requeridos:

```console
pip install -r requirements.txt
```

## Autoencoder
Es necesario generar un archivo de configuración .json, puede usar como ejemplo autoencoder.json que se encuentra en el directory config.

```console
python autoencoder.py --config_json ./config/autoencoder.json
```

## Denoising Autoencoder
Es necesario generar un archivo de configuración .json, puede usar como ejemplo autoencoder_denoising.json que se encuentra en el directory config.

```console
python autoencoder_denoising.py --config_json ./config/autoencoder_denoising.json
```

## VAE: Auto encoder variacional
Es necesario generar un archivo de configuración .json, puede usar como ejemplo autoencoder_vae.json que se encuentra en el directory config.

```console
python autoencoder_vae.py --config_json ./config/autoencoder_vae.json
```