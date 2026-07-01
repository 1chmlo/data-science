"""Genera un sitio web estatico para explorar los resultados del pipeline.

Escanea `resultados/{region,hospital}/*/metricas.json`, copia los graficos a
`web/assets/` con nombres ASCII y vuelca todo a `web/data.js` (variable global
`window.RESULTS`) para que `web/index.html` funcione con doble clic (sin servidor
ni fetch, evitando problemas de CORS en file://).

    python -m forecasting.build_web --results resultados --out web
"""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from pathlib import Path

_LEVELS = ("region", "hospital")


def _slug(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return "".join(c if c.isalnum() else "_" for c in norm).strip("_")


def _display(level: str, folder: str) -> str:
    return folder.replace("_", " ") if level == "region" else folder


def _record(level: str, d: Path, assets: Path) -> dict | None:
    meta_path = d / "metricas.json"
    if not meta_path.exists():
        return None
    m = json.loads(meta_path.read_text(encoding="utf-8"))

    slug = _slug(f"{level}_{d.name}")
    imgs: dict[str, str] = {}
    for key, fname in (("forecast", "forecast_vs_real.png"),
                       ("residuos", "residuos_diagnostico.png")):
        src = d / "graficos" / fname
        if src.exists():
            dst = assets / f"{slug}_{key}.png"
            shutil.copyfile(src, dst)
            imgs[key] = f"assets/{dst.name}"

    saved = m.get("modelo_guardado")
    orders = m.get("orders", {})
    return {
        "id": m.get("series_id", d.name),
        "level": level,
        "display": _display(level, d.name),
        "best_model": m.get("best_model"),
        "saved_model": saved,
        "order": orders.get(saved),
        "n_weeks": m.get("n_weeks"),
        "n_folds": m.get("n_folds"),
        "ic": m.get("ic_nivel"),
        "metrics": m.get("metrics_por_modelo", {}),
        "future": m.get("prediccion_futura"),
        "future_note": m.get("nota_futuro"),
        "diag": m.get("diagnostico_residuos", {}).get("interpretacion", {}),
        "warnings": m.get("warnings", []),
        "img_forecast": imgs.get("forecast"),
        "img_residuos": imgs.get("residuos"),
    }


def build(results_dir: Path, out_dir: Path) -> list[dict]:
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for level in _LEVELS:
        base = results_dir / level
        if not base.exists():
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            rec = _record(level, d, assets)
            if rec is not None:
                records.append(rec)

    payload = "window.RESULTS = " + json.dumps(records, ensure_ascii=False) + ";\n"
    (out_dir / "data.js").write_text(payload, encoding="utf-8")
    return records


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Genera el sitio web de resultados.")
    p.add_argument("--results", default="resultados", type=Path)
    p.add_argument("--out", default="web", type=Path)
    args = p.parse_args(argv)

    records = build(args.results, args.out)
    by_level: dict[str, int] = {}
    for r in records:
        by_level[r["level"]] = by_level.get(r["level"], 0) + 1
    print(f"Series volcadas: {len(records)} -> {args.out / 'data.js'}")
    print("por nivel:", by_level)
    print(f"Abre: {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
