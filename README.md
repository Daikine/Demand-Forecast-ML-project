# Нейросетевая модель для прогнозирования спроса на товары (MVP)

Проект сделан **строго по учебному кейсу**:

- **Спринт 1:** генерация синтетических данных + EDA
- **Спринт 2:** baseline модели + метрики
- **Спринт 3:** нейросеть **PyTorch LSTM** + метрики
- **Спринт 4:** MVP сервис на **Streamlit**
- **Спринт 5:** финализация: графики для презентации

## Структура

- `data/generate_data.py` — генерация синтетики (`data/sales.csv`)
- `notebooks/01_eda.ipynb` — EDA
- `src/evaluate_baselines.py` — baseline метрики (MAE/MSE/RMSE/MAPE)
- `src/train_torch.py` — обучение LSTM и сохранение артефактов
- `src/report.py` — графики для презентации
- `streamlit_app.py` — веб-интерфейс MVP

## Установка

```bash
pip install -r requirements.txt
```

## Milestone 1 — Данные и анализ

```bash
python data/generate_data.py --output data/sales.csv --days 1100 --skus 5 --seed 42
```

Открой EDA ноутбук:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Milestone 2 — Baseline

```bash
python -m src.evaluate_baselines --data data/sales.csv --lookback 30 --test-days 120
```

Результаты:
- `artifacts/baseline_metrics.csv`
- `artifacts/metrics_baselines.json`

## Milestone 3 — Нейросеть (PyTorch LSTM)

```bash
python -m src.train_torch --data data/sales.csv --lookback 28 --horizon 14 --test-days 120 --val-days 60 --epochs 40 --device cpu
```

Результаты:
- `artifacts/<SKU>/model.pt`
- `artifacts/<SKU>/feature_scaler.joblib`
- `artifacts/<SKU>/target_scaler.joblib`
- `artifacts/<SKU>/metrics_nn.json`
- `artifacts/metrics_nn_all.json`

## Milestone 5 — Графики для презентации

```bash
python -m src.report --data data/sales.csv --sku SKU_01
```

Результаты:
- `reports/figures/compare_mae.png`
- `reports/figures/example_series.png`

## Milestone 4 — MVP сервис (Streamlit)

```bash
python -m streamlit run streamlit_app.py
```

### Деплой на Streamlit Community Cloud

При создании приложения выбирай:
- Repository: твой GitHub repo
- Branch: `main`
- Main file path: `streamlit_app.py`

## CSV формат

Минимум:
- `date` (YYYY-MM-DD)
- `sales` (целое)

Опционально:
- `price`, `promo`, `discount_pct`, `is_weekend`, `is_holiday`

## Исправление ошибки SKU_01

В исходной версии у `SKU_01` была завышенная ошибка нейросети: `WAPE = 104.65%`. Причина — `ReLU` на выходе модели применялся к стандартизированному `log1p(sales)`. В таком пространстве нормальные значения могут быть отрицательными, а `ReLU` обрезал их до нуля и завышал прогноз для товаров с небольшим спросом.

Исправлено:
- выход LSTM теперь линейный, без `ReLU`;
- loss заменён на `SmoothL1Loss` в стандартизированном log-space;
- калибровка считается по validation, а не по test;
- `hidden_size` сохраняется и загружается корректно, а Streamlit дополнительно умеет восстановить размер модели из весов.

После переобучения `SKU_01`: `MAE = 5.04`, `RMSE = 6.15`, `WAPE = 25.73%`.
