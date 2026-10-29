# Лабораторная работа № 4
# Визуализация данных в Python: matplotlib, seaborn, plotly
# Датасет: Telco Customer Churn
#
# Перед запуском:
# 1. Установите библиотеки: pip install pandas matplotlib seaborn plotly
# 2. Запустите файл из папки с CSV-файлом либо разрешите загрузку датасета из интернета.

import sys
import subprocess
from pathlib import Path

# Проверяем наличие необходимых библиотек.
for module, package in [
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
]:
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    display = print

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 24)
pd.set_option("display.width", 120)

# Единая цветовая схема: синий — оставшиеся клиенты, красный — ушедшие.
CHURN_COLORS = {"No": "#4C72B0", "Yes": "#C44E52"}

print("pandas:", pd.__version__)
print("seaborn:", sns.__version__)
print("plotly:", plotly.__version__)




# ============================================================================
# 1. Загрузка данных
# ============================================================================

CSV_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
CSV_MIRROR = "https://huggingface.co/datasets/mnemoraorg/telco-churn-7k/resolve/main/telco_churn_7k.csv"
LOCAL_CANDIDATES = [
    Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    Path("Telco-Customer-Churn.csv"),
    Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    Path("data") / "Telco-Customer-Churn.csv",
    Path("/content") / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

df = None
source = None

for url in [CSV_URL, CSV_MIRROR]:
    try:
        df = pd.read_csv(url)
        source = url
        break
    except Exception as err:
        print(f"Недоступно: {url} ({type(err).__name__})")

if df is None:
    for path in LOCAL_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            source = str(path)
            break

if df is None:
    raise FileNotFoundError(
        "Датасет не найден. Скачайте WA_Fn-UseC_-Telco-Customer-Churn.csv "
        "(Kaggle: blastchar/telco-customer-churn) и положите рядом с ноутбуком."
    )

print("Источник:", source)
print("Строк и столбцов:", df.shape)


# ============================================================================
# Просмотр первых строк данных
# ============================================================================

display(df.head())


# ============================================================================
# Общая информация о типах данных и пропусках
# ============================================================================

df.info()


# ============================================================================
# 2. Приведение TotalCharges к числовому типу
# ============================================================================

print("Тип до приведения:", df["TotalCharges"].dtype)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Тип после приведения:", df["TotalCharges"].dtype)
print("Пропусков появилось:", int(df["TotalCharges"].isna().sum()))

display(df.loc[df["TotalCharges"].isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])


# ============================================================================
# Создание числового признака оттока
# ============================================================================

df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

print("Общая доля оттока:", round(df["ChurnFlag"].mean(), 3))
display(df[["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]].describe().round(2))


# ============================================================================
# 3. Частотное сравнение: matplotlib
# ============================================================================

fig, ax = plt.subplots()

ax.hist(df["MonthlyCharges"], bins=30, color="#4C72B0", edgecolor="white")

ax.set_title("Распределение месячного платежа (matplotlib)")
ax.set_xlabel("Месячный платёж, $")
ax.set_ylabel("Количество клиентов")

plt.show()


# ============================================================================
# 3. Частотное сравнение: seaborn
# ============================================================================

ax = sns.histplot(data=df, x="MonthlyCharges", bins=30, kde=True, color="#4C72B0")

ax.set_title("Распределение месячного платежа (seaborn)")
ax.set_xlabel("Месячный платёж, $")
ax.set_ylabel("Количество клиентов")

plt.show()


# ============================================================================
# Сравнение распределения платежа по признаку оттока
# ============================================================================

ax = sns.histplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    bins=30,
    element="step",
    stat="density",
    common_norm=False,
    palette=CHURN_COLORS,
)

ax.set_title("Месячный платёж: ушедшие и оставшиеся клиенты (seaborn)")
ax.set_xlabel("Месячный платёж, $")
ax.set_ylabel("Плотность")

plt.show()


# ============================================================================
# 3. Частотное сравнение: plotly
# ============================================================================

fig = px.histogram(
    df,
    x="MonthlyCharges",
    color="Churn",
    nbins=30,
    barmode="overlay",
    opacity=0.7,
    color_discrete_map=CHURN_COLORS,
    title="Распределение месячного платежа (plotly)",
    labels={"MonthlyCharges": "Месячный платёж, $", "Churn": "Отток"},
)

fig.update_layout(
    yaxis_title="Количество клиентов",
    bargap=0.05,
    width=900,
    height=420,
    legend_title_text="Отток",
)

fig.show()


# ============================================================================
# Дополнительное задание 1. Распределение TotalCharges в трёх библиотеках
# ============================================================================
# TotalCharges — сумма платежей за весь период обслуживания.
# В отличие от MonthlyCharges, показатель зависит и от ежемесячного тарифа,
# и от длительности обслуживания. Поэтому распределение обычно имеет широкий
# диапазон, а у новых клиентов могут быть нулевые/очень маленькие суммы.

total = df["TotalCharges"].dropna()

# Matplotlib
fig, ax = plt.subplots()
ax.hist(total, bins=35, color="#8172B2", edgecolor="white")
ax.set_title("Распределение TotalCharges (matplotlib)")
ax.set_xlabel("Суммарные платежи, $")
ax.set_ylabel("Количество клиентов")
plt.tight_layout()
plt.show()

# Seaborn
fig, ax = plt.subplots()
sns.histplot(data=df, x="TotalCharges", bins=35, kde=True, color="#8172B2", ax=ax)
ax.set_title("Распределение TotalCharges (seaborn)")
ax.set_xlabel("Суммарные платежи, $")
ax.set_ylabel("Количество клиентов")
plt.tight_layout()
plt.show()

# Plotly Express
fig = px.histogram(
    df.dropna(subset=["TotalCharges"]),
    x="TotalCharges",
    color="Churn",
    nbins=35,
    barmode="overlay",
    opacity=0.65,
    color_discrete_map=CHURN_COLORS,
    title="Распределение TotalCharges (plotly)",
    labels={"TotalCharges": "Суммарные платежи, $", "Churn": "Отток"},
)
fig.update_layout(yaxis_title="Количество клиентов", width=900, height=420)
fig.show()

print(
    "Вывод: MonthlyCharges показывает текущую ежемесячную стоимость и обычно "
    "имеет ограниченный диапазон. TotalCharges накапливается за весь срок "
    "обслуживания, поэтому имеет намного больший разброс и зависит от tenure. "
    "У клиентов с коротким сроком обслуживания суммарные платежи невелики."
)


# ============================================================================
# 4. Сравнение по способу оплаты: сводная таблица
# ============================================================================

payment_counts = df["PaymentMethod"].value_counts()
payment_share = df["PaymentMethod"].value_counts(normalize=True).mul(100).round(1)

summary = pd.DataFrame({"Клиентов": payment_counts, "Доля, %": payment_share})
summary["Доля оттока, %"] = (
    df.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).round(1)
)

display(summary)


# ============================================================================
# 4.1. Столбчатая диаграмма seaborn
# ============================================================================

fig, ax = plt.subplots(figsize=(9, 4))

sns.countplot(
    data=df,
    y="PaymentMethod",
    hue="Churn",
    order=payment_counts.index,
    palette=CHURN_COLORS,
    ax=ax,
)

ax.set_title("Клиенты по способу оплаты и оттоку (seaborn)")
ax.set_xlabel("Количество клиентов")
ax.set_ylabel("")
ax.legend(title="Отток")

plt.tight_layout()
plt.show()


# ============================================================================
# 4.2. Столбчатая диаграмма plotly
# ============================================================================

fig = px.histogram(
    df,
    y="PaymentMethod",
    color="Churn",
    barmode="group",
    color_discrete_map=CHURN_COLORS,
    category_orders={"PaymentMethod": list(payment_counts.index)},
    title="Клиенты по способу оплаты (plotly)",
    labels={"PaymentMethod": "Способ оплаты", "Churn": "Отток"},
)

fig.update_layout(xaxis_title="Количество клиентов", width=900, height=420, legend_title_text="Отток")
fig.show()


# ============================================================================
# Доля оттока по способу оплаты
# ============================================================================

churn_by_payment = (
    df.groupby("PaymentMethod", as_index=False)["ChurnFlag"]
    .mean()
    .assign(ChurnRate=lambda t: (t["ChurnFlag"] * 100).round(1))
    .sort_values("ChurnRate")
)

fig = px.bar(
    churn_by_payment,
    x="ChurnRate",
    y="PaymentMethod",
    orientation="h",
    text="ChurnRate",
    range_x=[0, 50],
    title="Доля оттока по способу оплаты, %",
    labels={"ChurnRate": "Доля оттока, %", "PaymentMethod": "Способ оплаты"},
)

fig.update_traces(marker_color="#C44E52", texttemplate="%{text}%", textposition="outside")
fig.update_layout(width=900, height=380)
fig.show()


# ============================================================================
# 5. Временное сравнение по сроку обслуживания
# ============================================================================

tenure_stats = (
    df.groupby("tenure")
    .agg(
        clients=("customerID", "count"),
        churn_rate=("ChurnFlag", "mean"),
        avg_monthly=("MonthlyCharges", "mean"),
    )
    .reset_index()
)

tenure_stats["churn_rate"] = (tenure_stats["churn_rate"] * 100).round(1)
tenure_stats["avg_monthly"] = tenure_stats["avg_monthly"].round(1)

display(tenure_stats.head())


# ============================================================================
# Дополнительное задание 3. Кривая оттока отдельно по каждому типу договора
# ============================================================================
# Важно: это срез данных по tenure, а не полноценное продольное наблюдение
# за одной и той же когортой клиентов. График показывает долю Churn="Yes"
# среди клиентов каждого типа договора с указанным tenure.

cohort_stats = (
    df.groupby(["tenure", "Contract"], as_index=False)
    .agg(
        clients=("customerID", "count"),
        churn_rate=("ChurnFlag", "mean"),
    )
)
cohort_stats["churn_rate"] = cohort_stats["churn_rate"] * 100

# Чтобы не делать выводы по очень маленьким группам, показываем линии только
# для точек, где в группе не менее 10 клиентов. Порог можно изменить.
MIN_GROUP_SIZE = 10
cohort_stats.loc[cohort_stats["clients"] < MIN_GROUP_SIZE, "churn_rate"] = float("nan")

fig = px.line(
    cohort_stats,
    x="tenure",
    y="churn_rate",
    color="Contract",
    markers=True,
    title="Кривая доли оттока по сроку обслуживания и типу договора",
    labels={
        "tenure": "Срок обслуживания, месяцев",
        "churn_rate": "Доля оттока, %",
        "Contract": "Тип договора",
    },
)
fig.update_layout(width=950, height=480, yaxis_range=[0, 100])
fig.show()

print(
    "Интерпретация: сравните линии трёх типов договора. Если значения для "
    "Month-to-month выше, это указывает на большую долю оттока в этой группе. "
    "Пропуски на графике означают, что в соответствующей точке было меньше "
    f"{MIN_GROUP_SIZE} клиентов и оценка могла быть нестабильной."
)


# ============================================================================
# График доли оттока по сроку обслуживания
# ============================================================================

fig = px.line(
    tenure_stats,
    x="tenure",
    y="churn_rate",
    markers=True,
    title="Доля оттока в зависимости от срока обслуживания",
    labels={"tenure": "Срок обслуживания, месяцев", "churn_rate": "Доля оттока, %"},
)

fig.update_traces(line_color="#C44E52")
fig.update_layout(width=900, height=420, yaxis_range=[0, 65])
fig.show()


# ============================================================================
# Доля оттока и численность клиентов на общей оси X
# ============================================================================

fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

top.plot(tenure_stats["tenure"], tenure_stats["churn_rate"], color="#C44E52")
top.set_ylabel("Доля оттока, %")
top.set_ylim(0, 65)
top.set_title("Отток и численность клиентов по сроку обслуживания")

bottom.fill_between(tenure_stats["tenure"], tenure_stats["clients"], color="#4C72B0", alpha=0.35)
bottom.plot(tenure_stats["tenure"], tenure_stats["clients"], color="#4C72B0")
bottom.set_ylabel("Клиентов, чел.")
bottom.set_xlabel("Срок обслуживания, месяцев")
bottom.set_ylim(0, None)

plt.tight_layout()
plt.show()


# ============================================================================
# 6. Корреляционное сравнение: pairplot
# ============================================================================

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

grid = sns.pairplot(
    df[numeric_cols + ["Churn"]].dropna(),
    hue="Churn",
    corner=True,
    palette=CHURN_COLORS,
    plot_kws={"alpha": 0.35, "s": 12, "edgecolor": "none"},
    height=2.4,
)

grid.figure.suptitle("Парные зависимости числовых признаков в разбивке по оттоку", y=1.02)
plt.show()


# ============================================================================
# Корреляционная матрица числовых признаков
# ============================================================================

display(df[numeric_cols].corr().round(2))


# ============================================================================
# Диаграмма рассеяния по сроку и суммарным платежам
# ============================================================================

fig = px.scatter(
    df.dropna(subset=["TotalCharges"]),
    x="tenure",
    y="TotalCharges",
    color="Churn",
    color_discrete_map=CHURN_COLORS,
    facet_col="Contract",
    opacity=0.5,
    hover_data=["MonthlyCharges", "InternetService", "PaymentMethod"],
    title="Срок обслуживания и суммарные платежи по типу договора",
    labels={
        "tenure": "Срок, месяцев",
        "TotalCharges": "Сумма платежей, $",
        "Churn": "Отток",
        "Contract": "Договор",
    },
)

fig.update_traces(marker_size=5)
fig.update_layout(width=980, height=420, legend_title_text="Отток")
fig.show()


# ============================================================================
# 7. Сравнение boxplot и violin plot
# ============================================================================

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn",
            palette=CHURN_COLORS, legend=False, ax=left)
left.set_title("Боксплот")
left.set_xlabel("Отток")
left.set_ylabel("Месячный платёж, $")

sns.violinplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn",
               palette=CHURN_COLORS, legend=False, inner="quartile", ax=right)
right.set_title("Violin plot")
right.set_xlabel("Отток")
right.set_ylabel("")

plt.tight_layout()
plt.show()


# ============================================================================
# Boxplot по типу договора
# ============================================================================

fig = px.box(
    df,
    x="Contract",
    y="MonthlyCharges",
    color="Churn",
    color_discrete_map=CHURN_COLORS,
    points="outliers",
    title="Месячный платёж по типу договора и оттоку (plotly, боксплот)",
    labels={"Contract": "Договор", "MonthlyCharges": "Месячный платёж, $", "Churn": "Отток"},
)

fig.update_layout(width=900, height=430, boxmode="group", legend_title_text="Отток")
fig.show()


# ============================================================================
# Violin plot по типу договора
# ============================================================================

fig = px.violin(
    df,
    x="Contract",
    y="MonthlyCharges",
    color="Churn",
    color_discrete_map=CHURN_COLORS,
    box=True,
    points=False,
    title="Месячный платёж по типу договора и оттоку (plotly, violin plot)",
    labels={"Contract": "Договор", "MonthlyCharges": "Месячный платёж, $", "Churn": "Отток"},
)

fig.update_layout(width=900, height=430, violinmode="group", legend_title_text="Отток")
fig.show()


# ============================================================================
# 8. Тепловая карта: способ оплаты × тип договора
# ============================================================================

# Количество клиентов в каждой комбинации PaymentMethod × Contract.
counts_pivot = pd.crosstab(df["PaymentMethod"], df["Contract"])

# Доля оттока в каждой комбинации.
churn_pivot = pd.pivot_table(
    df,
    index="PaymentMethod",
    columns="Contract",
    values="ChurnFlag",
    aggfunc="mean",
).mul(100).round(1)

display(counts_pivot)
display(churn_pivot)

# Показываем численность и долю оттока рядом, чтобы не трактовать проценты
# из малочисленных групп без осторожности.
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))

sns.heatmap(counts_pivot, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=left)
left.set_title("Численность группы, клиентов")
left.set_xlabel("Тип договора")
left.set_ylabel("Способ оплаты")

sns.heatmap(
    churn_pivot,
    annot=True,
    fmt=".1f",
    cmap="Reds",
    cbar=True,
    vmin=0,
    vmax=100,
    ax=right,
)
right.set_title("Доля оттока, %")
right.set_xlabel("Тип договора")
right.set_ylabel("")

plt.tight_layout()
plt.show()

# Выводим комбинации с высокой долей оттока, но малым количеством клиентов.
long_summary = (
    df.groupby(["PaymentMethod", "Contract"], as_index=False)
    .agg(clients=("customerID", "count"), churn_rate=("ChurnFlag", "mean"))
)
long_summary["churn_rate_pct"] = (long_summary["churn_rate"] * 100).round(1)

SMALL_GROUP_LIMIT = 30
HIGH_CHURN_LIMIT = 50
suspicious = long_summary[
    (long_summary["clients"] < SMALL_GROUP_LIMIT)
    & (long_summary["churn_rate_pct"] >= HIGH_CHURN_LIMIT)
].sort_values(["churn_rate_pct", "clients"], ascending=[False, True])

print(
    f"Комбинации с численностью менее {SMALL_GROUP_LIMIT} клиентов "
    f"и долей оттока не ниже {HIGH_CHURN_LIMIT}%:"
)
if suspicious.empty:
    print("Таких комбинаций по выбранным порогам не обнаружено.")
else:
    display(suspicious[["PaymentMethod", "Contract", "clients", "churn_rate_pct"]])

print(
    "Интерпретация: высокий процент в малочисленной группе не следует считать "
    "надёжным общим законом. Сначала нужно проверить число клиентов и "
    "количество ушедших; доля может резко меняться из-за нескольких наблюдений."
)

# Интерактивная тепловая карта Plotly.
fig = px.imshow(
    churn_pivot,
    text_auto=True,
    color_continuous_scale="Reds",
    zmin=0,
    zmax=100,
    aspect="auto",
    title="Доля оттока, %: способ оплаты × тип договора (plotly)",
    labels={"x": "Тип договора", "y": "Способ оплаты", "color": "Отток, %"},
)
fig.update_layout(width=850, height=430)
fig.show()


# ============================================================================
# Дополнительное задание 4. Намеренно испорченный график
# ============================================================================
# Здесь ось Y обрезана, подписи категорий скрыты, а цвета сделаны кричащими.
# Это пример плохой визуализации, а не график для реальных выводов.

bad_data = (
    df.groupby("PaymentMethod", as_index=False)["ChurnFlag"]
    .mean()
    .assign(churn_pct=lambda x: x["ChurnFlag"] * 100)
)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(
    bad_data["PaymentMethod"],
    bad_data["churn_pct"],
    color=["#FF00FF", "#00FF00", "#FFFF00", "#00FFFF"][:len(bad_data)],
)
ax.set_ylim(35, 50)  # Обрезанная ось усиливает визуальные различия.
ax.set_xticks([])    # Убраны подписи категорий.
ax.set_title("ПЛОХОЙ ПРИМЕР: доля оттока")
ax.set_ylabel("")    # Убрана подпись оси.
plt.tight_layout()
plt.show()

print(
    "Какой неверный вывод может сделать читатель: различия между способами "
    "оплаты кажутся огромными, хотя ось начинается не с нуля; при этом из-за "
    "отсутствия подписей невозможно понять, какая категория соответствует "
    "каждому столбцу. Яркие цвета дополнительно отвлекают и создают ложный акцент."
)


# ============================================================================
# 9. Итоги
# ============================================================================
print("\
Анализ завершён.")
print("Основные направления анализа: распределение месячных платежей, способ оплаты,")
print("срок обслуживания, числовые зависимости, тип договора и доля оттока.")
print("Интерактивные графики Plotly открываются в окне вывода или браузере.")
