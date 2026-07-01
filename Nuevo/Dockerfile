# syntax=docker/dockerfile:1

# --- Etapa 1: generar el sitio estatico (data.js + assets) desde resultados/ ---
# build_web solo usa la biblioteca estandar, asi que no hace falta instalar nada.
FROM python:3.12-slim AS build
WORKDIR /app
COPY forecasting/ ./forecasting/
COPY resultados/ ./resultados/
COPY web/index.html ./web/index.html
RUN python -m forecasting.build_web --results resultados --out web

# --- Etapa 2: servir el sitio con nginx ---
FROM nginx:alpine
COPY --from=build /app/web /usr/share/nginx/html
EXPOSE 80
