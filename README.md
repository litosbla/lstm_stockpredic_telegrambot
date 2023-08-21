# lstm_stockpredic_telegrambot
bot de telegram que usa lstm para predecir el par EUR/USD
## Requisitos

- Sistema operativo: Windows
- [MetaTrader5](https://www.metatrader5.com/es) debe estar instalado en tu sistema.

## Instalación

1. Clona o descarga este repositorio.
2. Crea y activa un entorno virtual (opcional, pero recomendado).
3. Instala las dependencias usando el siguiente comando:

```bash
pip install -r requirements.txt
```
## Configuración
1.Modifica un archivo llamado **config.json**  agregandole el token de Telegram al archivo. si no sabes como crear un bot de telegram visita el siguiente sitio  [Api-Telegram](https://core.telegram.org/bots/api)

```json
{
  "telegram_token": "TU_TOKEN_AQUI"
}
```
## Uso
El bot tiene 4 funciones
1. **/start** ---> Inicio
2. **/predict_chart ---> se conecta a Metatrader5, toma los datos actuales y predice las siguiente 5 velas, manda como respuesta una grafica y una opcion de compra o venta.
Nota: Si no existe ningun modelo entrenado en la raiz, el entrenará uno de lo contrario hara las predicciones en base al modelo guardado
3. **/entrenar** ---> entrena el modelo, esta tarea puede demorar un poco
4. **/predict_chart_hourly** ---> Si no existe ningun modelo entrenado en la raiz, el automaticamente entrenará y hara predicciones cada hora sin necesidad de volverle a presionar otro comando

## Contribuciones
Si deseas contribuir al proyecto, sigue los siguientes pasos:

1.Haz un fork de este repositorio.
2.Crea una rama para tu función/feature: git checkout -b nueva-funcion.
3.Realiza tus cambios y commitea: git commit -am 'Agregada nueva función'.
4.Sube los cambios a tu repositorio: git push origin nueva-funcion.
5.Abre un Pull Request explicando tus cambios.
