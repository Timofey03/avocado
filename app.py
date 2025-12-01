import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta
import random

# ==========================================
# 1. НАСТРОЙКИ СТРАНИЦЫ
# ==========================================
st.set_page_config(
    page_title="Avocado Price Forecaster",
    page_icon="🥑",
    layout="wide"
)

# Стили
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    h1 {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

st.title("🥑 Прогноз цен на авокадо")
st.markdown("""
Приложение использует ансамбль **RandomForest + CatBoost**.
Прогноз строится рекурсивно: модель предсказывает следующую неделю, запоминает её и использует для следующего шага.
Шум теперь всегда применяется к прогнозу (процентный, ±5%, воспроизводимый).
""")

# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ И МОДЕЛИ
# ==========================================
@st.cache_resource
def load_artifact():
    try:
        return joblib.load("avocado_artifact.pkl")
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("avocado.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    except FileNotFoundError:
        return None

artifact = load_artifact()
df = load_data()

if artifact is None or df is None:
    st.error("🚨 Файлы `avocado.csv` или `avocado_artifact.pkl` не найдены.")
    st.info("Поместите файлы в ту же папку, что и `app.py`.")
    st.stop()

rf_model = artifact["rf_model"]
cb_model = artifact["catboost_model"]
feature_order = artifact["features"]

st.sidebar.success("✅ Данные и модель загружены")

# ==========================================
# 3. БОКОВАЯ ПАНЕЛЬ
# ==========================================
st.sidebar.header("⚙️ Параметры прогноза")
regions = sorted(df['region'].unique())
default_ix = regions.index("TotalUS") if "TotalUS" in regions else 0
selected_region = st.sidebar.selectbox("Регион:", regions, index=default_ix)

types = sorted(df['type'].unique())
selected_type = st.sidebar.selectbox("Тип авокадо:", types, index=0)

horizon_map = {
    "1 месяц (4 недели)": 4,
    "2 месяца (8 недель)": 8,
    "1 квартал (13 недель)": 13,
    "Полгода (26 недель)": 26,
    "1 год (52 недели)": 52
}
horizon_label = st.sidebar.selectbox("Горизонт прогноза:", list(horizon_map.keys()), index=2)
weeks_ahead = horizon_map[horizon_label]

st.sidebar.markdown("---")

# ==========================================
# 4. ФУНКЦИЯ ПРОГНОЗА (ШУМ ВСЕГДА ВКЛЮЧЁН)
# ==========================================
def recursive_forecast(weeks, region, type_name):
    history = df[(df['region'] == region) & (df['type'] == type_name)].sort_values('Date').tail(52).copy()
    if len(history) < 10:
        return None, None

    work_history = history.tail(20).copy()

    # Оценка процентной волатильности (историческая)
    price_pct_changes = work_history['AveragePrice'].pct_change().dropna()
    vol_pct = price_pct_changes.std()
    if pd.isna(vol_pct) or vol_pct < 0.005:
        vol_pct = 0.02  # 2% по умолчанию

    # Фиксируем seed для воспроизводимости шумов
    random.seed(42)

    predictions = []
    current_last_date = work_history['Date'].iloc[-1]

    progress_text = "Вычисление прогноза..."
    my_bar = st.progress(0, text=progress_text)

    for i in range(weeks):
        next_date = current_last_date + timedelta(weeks=1)

        # Сбор признаков
        row = {}
        row['year'] = next_date.year
        row['month'] = next_date.month
        row['weekofyear'] = next_date.isocalendar().week
        row['quarter'] = next_date.quarter

        # Лаги
        row['lag1'] = work_history['AveragePrice'].iloc[-1]
        row['lag2'] = work_history['AveragePrice'].iloc[-2]
        row['lag3'] = work_history['AveragePrice'].iloc[-3]
        row['lag4'] = work_history['AveragePrice'].iloc[-4]

        # Скользящие
        last_4 = work_history['AveragePrice'].tail(4)
        row['rolling_mean_4'] = last_4.mean()
        row['rolling_std_4'] = last_4.std() if len(last_4) > 1 else 0

        # Категории
        row['region'] = region
        row['type'] = type_name

        X_input = pd.DataFrame([row])
        for col in feature_order:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_order]

        # Прогноз модели
        pred_rf = rf_model.predict(X_input)[0]
        pred_cb = cb_model.predict(X_input)[0]
        next_price = (pred_rf + pred_cb) / 2

        # --- Автоматический процентный шум (всегда включён) ---
        noise_factor = random.gauss(0, vol_pct)
        noise_factor = max(min(noise_factor, 0.05), -0.05)  # ограничение ±5%
        next_price *= (1 + noise_factor)
        if next_price < 0.5:
            next_price = 0.5

        # Обновление истории (чтобы лаги включали прогноз)
        new_row = {'Date': next_date, 'AveragePrice': next_price, 'region': region, 'type': type_name}
        work_history = pd.concat([work_history, pd.DataFrame([new_row])], ignore_index=True)

        predictions.append({'Date': next_date, 'Predicted_Price': next_price})
        current_last_date = next_date
        my_bar.progress((i + 1) / weeks, text=progress_text)

    my_bar.empty()
    return pd.DataFrame(predictions), history

# ==========================================
# 5. ОСНОВНОЙ ИНТЕРФЕЙС
# ==========================================
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 🚀 Управление")
    run_btn = st.button("Рассчитать прогноз", type="primary", use_container_width=True)
    st.info(f"Будет построен прогноз на **{weeks_ahead} недель** вперед.")

if run_btn:
    with st.spinner('Модели работают...'):
        # шум теперь применяется всегда — передаём только необходимые параметры
        forecast_df, history_df = recursive_forecast(weeks_ahead, selected_region, selected_type)

    if forecast_df is None:
        st.error(f"Недостаточно данных для региона {selected_region}!")
    else:
        st.markdown("---")
        last_hist_price = history_df['AveragePrice'].iloc[-1]
        last_pred_price = forecast_df['Predicted_Price'].iloc[-1]
        min_pred = forecast_df['Predicted_Price'].min()
        max_pred = forecast_df['Predicted_Price'].max()
        delta = last_pred_price - last_hist_price

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Цена сейчас", f"${last_hist_price:.2f}")
        m2.metric("Цена в конце", f"${last_pred_price:.2f}", delta=f"{delta:.2f}")
        m3.metric("Мин. прогноз", f"${min_pred:.2f}")
        m4.metric("Макс. прогноз", f"${max_pred:.2f}")

        # График
        st.subheader(f"График: {selected_region} ({selected_type})")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(history_df['Date'], history_df['AveragePrice'], label='История', linewidth=2, marker='o', markersize=4)
        ax.plot(forecast_df['Date'], forecast_df['Predicted_Price'], label='Прогноз (с шумом)', linewidth=2, linestyle='--', marker='o', markersize=4)
        ax.plot([history_df['Date'].iloc[-1], forecast_df['Date'].iloc[0]],
                [history_df['AveragePrice'].iloc[-1], forecast_df['Predicted_Price'].iloc[0]],
                linestyle='--', linewidth=2)
        ax.set_title(f"Прогноз цены на {weeks_ahead} недель", fontsize=14)
        ax.set_ylabel("Цена ($)", fontsize=12)
        ax.set_xlabel("Дата", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Таблица и скачивание
        st.markdown("---")
        col_table, col_dl = st.columns([2, 1])
        with col_table:
            with st.expander("📄 Посмотреть таблицу данных"):
                st.dataframe(forecast_df.style.format({"Predicted_Price": "${:.2f}"}))
        with col_dl:
            st.write("### Экспорт")
            csv_data = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать прогноз (CSV)",
                data=csv_data,
                file_name=f"forecast_{selected_region}_{selected_type}.csv",
                mime="text/csv",
                type="primary"
            )

else:
    st.info("👈 Выберите параметры слева и нажмите кнопку **'Рассчитать прогноз'**.")
