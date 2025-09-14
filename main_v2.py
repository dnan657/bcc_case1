import os
import glob
import math
import json
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

DATA_DIR = Path("./data")
OUT_DIR = Path("./")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Категории и константы
TRAVEL_CATS = {"Такси", "Путешествия", "Отели"}
ONLINE_CATS = {"Играем дома", "Смотрим дома", "Кино", "Едим дома"}
JPR_CATS = {"Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"}

MONTHS_RU_GEN = {
    1: "январе",
    2: "феврале",
    3: "марте",
    4: "апреле",
    5: "мае",
    6: "июне",
    7: "июле",
    8: "августе",
    9: "сентябре",
    10: "октябре",
    11: "ноябре",
    12: "декабре",
}

def fmt_kzt(x: float, decimals: int = 0) -> str:
    if x is None or np.isnan(x):
        return "0 ₸"
    q = round(float(x), decimals)
    s = f"{q:,.{decimals}f}"
    s = s.replace(",", " ").replace(".", ",")
    return f"{s} ₸"

def fmt_num(x: float, decimals: int = 0) -> str:
    q = round(float(x), decimals)
    s = f"{q:,.{decimals}f}"
    s = s.replace(",", " ").replace(".", ",")
    return s

def safe_read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", parse_dates=parse_dates)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8", parse_dates=parse_dates)

def month_range_last(df_dates: pd.Series) -> tuple[int, int]:
    if df_dates.empty:
        # fallback — текущий месяц
        now = datetime.now()
        return now.year, now.month
    last = pd.to_datetime(df_dates.max())
    return last.year, last.month

def filter_month(df: pd.DataFrame, year: int, month: int, date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    return d[(d[date_col].dt.year == year) & (d[date_col].dt.month == month)]

def top_categories(spend_by_cat: pd.Series, n=3) -> list[str]:
    if spend_by_cat.empty:
        return []
    return list(spend_by_cat.sort_values(ascending=False).head(n).index)

def union_spend(spend_map: pd.Series, cats: set[str]) -> float:
    if spend_map.empty or not cats:
        return 0.0
    return float(spend_map.reindex(cats, fill_value=0.0).sum())

def detect_fx_currency(trans_month: pd.DataFrame, trans_all: pd.DataFrame, trf_month: pd.DataFrame) -> str | None:
    # 1) по транзакциям в валюте != KZT
    if not trans_month.empty:
        cur_counts = (trans_month.assign(currency=trans_month["currency"].fillna("KZT"))
                      .loc[trans_month["currency"].ne("KZT"), "currency"]
                      .value_counts())
        if not cur_counts.empty:
            return cur_counts.idxmax()
    # 2) по типам fx_buy/fx_sell в переводах — попробуем взять currency колонки
    if not trf_month.empty:
        fx = trf_month[trf_month["type"].isin(["fx_buy", "fx_sell"])]
        if not fx.empty:
            cur_counts = fx["currency"].fillna("").replace("", np.nan).dropna().value_counts()
            if not cur_counts.empty and cur_counts.index[0] != "KZT":
                return cur_counts.index[0]
    # 3) fallback
    return None

def benefit_travel(spend_map: pd.Series) -> float:
    travel_spend = union_spend(spend_map, TRAVEL_CATS)
    return 0.04 * travel_spend

def benefit_premium(total_spend: float, spend_map: pd.Series, avg_balance: float,
                    atm_withdrawal_amt: float, card_p2p_amt: float) -> float:
    # Базовый процент по остатку
    if avg_balance >= 6_000_000:
        base = 0.04
    elif avg_balance >= 1_000_000:
        base = 0.03
    else:
        base = 0.02
    base_cashback = base * total_spend

    # Повышенные категории до 4%
    extra_pct = max(0.0, 0.04 - base)
    jpr_spend = union_spend(spend_map, JPR_CATS)
    extra_cashback = extra_pct * jpr_spend

    # Лимит кешбэка
    cashback = min(base_cashback + extra_cashback, 100_000.0)

    # Экономия комиссий (эвристика)
    atm_saving = min(atm_withdrawal_amt, 3_000_000.0) * 0.003  # 0,3% до 3 млн
    transfers_saving = card_p2p_amt * 0.002  # 0,2% на переводы по РК
    return cashback + atm_saving + transfers_saving

def benefit_credit(spend_map: pd.Series) -> float:
    if spend_map.empty:
        return 0.0
    cats_top3 = set(top_categories(spend_map, n=3))
    union_cats = cats_top3.union(ONLINE_CATS)
    union_amount = union_spend(spend_map, union_cats)
    return 0.10 * union_amount

def benefit_fx(trans_month: pd.DataFrame, trf_month: pd.DataFrame) -> float:
    # FX объём = суммы fx_buy/fx_sell + транзакции в валюте != KZT
    fx_trf = 0.0
    if not trf_month.empty:
        fx = trf_month[trf_month["type"].isin(["fx_buy", "fx_sell"])]
        fx_trf = float(fx["amount"].sum()) if not fx.empty else 0.0
    non_kzt = 0.0
    if not trans_month.empty:
        non = trans_month[trans_month["currency"].ne("KZT")]
        non_kzt = float(non["amount"].sum()) if not non.empty else 0.0
    fx_volume = fx_trf + non_kzt
    return 0.005 * fx_volume  # ~0,5% экономия

def choose_deposit_type(avg_balance: float, fx_active: bool, has_topups: bool, low_withdrawals: bool) -> str:
    if fx_active:
        return "Депозит Мультивалютный"
    if has_topups:
        return "Депозит Накопительный"
    if low_withdrawals:
        return "Депозит Сберегательный"
    # по умолчанию — накопительный, как более гибкий
    return "Депозит Накопительный"

def deposit_rate(product: str) -> float:
    if product == "Депозит Сберегательный":
        return 0.165
    if product == "Депозит Накопительный":
        return 0.155
    if product == "Депозит Мультивалютный":
        return 0.145
    return 0.0

def benefit_deposit(product: str, avg_balance: float) -> float:
    rate = deposit_rate(product)
    return (rate / 12.0) * max(0.0, avg_balance)

def detect_need_cash_loan(avg_balance: float, inflows: float, outflows: float) -> bool:
    # Явная потребность: значимый отрицательный поток и низкий остаток
    return (outflows - inflows) > 100_000.0 and avg_balance < 150_000.0

def summarize_month(trans_m: pd.DataFrame, trf_m: pd.DataFrame) -> dict:
    spend_total = float(trans_m["amount"].sum()) if not trans_m.empty else 0.0
    spend_by_cat = pd.Series(dtype=float)
    if not trans_m.empty:
        spend_by_cat = trans_m.groupby("category")["amount"].sum()

    taxi_cnt = int(trans_m.loc[trans_m["category"] == "Такси"].shape[0]) if not trans_m.empty else 0
    taxi_sum = float(trans_m.loc[trans_m["category"] == "Такси", "amount"].sum()) if not trans_m.empty else 0.0
    restaurants_sum = float(trans_m.loc[trans_m["category"] == "Кафе и рестораны", "amount"].sum()) if not trans_m.empty else 0.0
    online_sum = union_spend(spend_by_cat, ONLINE_CATS)

    atm_withdrawal = 0.0
    card_out = 0.0
    p2p_out = 0.0
    utilities_out = 0.0
    inflows = 0.0
    if not trf_m.empty:
        atm_withdrawal = float(trf_m.loc[trf_m["type"] == "atm_withdrawal", "amount"].sum())
        card_out = float(trf_m.loc[trf_m["type"] == "card_out", "amount"].sum())
        p2p_out = float(trf_m.loc[trf_m["type"] == "p2p_out", "amount"].sum())
        utilities_out = float(trf_m.loc[trf_m["type"] == "utilities_out", "amount"].sum())
        inflows = float(trf_m.loc[trf_m["direction"] == "in", "amount"].sum())

    return {
        "spend_total": spend_total,
        "spend_by_cat": spend_by_cat,
        "taxi_cnt": taxi_cnt,
        "taxi_sum": taxi_sum,
        "restaurants_sum": restaurants_sum,
        "online_sum": online_sum,
        "atm_withdrawal": atm_withdrawal,
        "card_p2p": card_out + p2p_out,
        "inflows": inflows,
        "utilities_out": utilities_out,
    }

def analyze_client(client_row: pd.Series) -> dict:
    client_code = int(client_row["client_code"])
    name = str(client_row["name"])
    status = str(client_row["status"])
    city = str(client_row["city"])
    avg_balance = float(client_row["avg_monthly_balance_KZT"])

    f_tr = DATA_DIR / f"client_{client_code}_transactions_3m.csv"
    f_tf = DATA_DIR / f"client_{client_code}_transfers_3m.csv"

    trans_all = safe_read_csv(f_tr, parse_dates=["date"])
    trans_all = trans_all if not trans_all.empty else pd.DataFrame(columns=["date","category","amount","currency"])
    transfers_all = safe_read_csv(f_tf, parse_dates=["date"])
    transfers_all = transfers_all if not transfers_all.empty else pd.DataFrame(columns=["date","type","direction","amount","currency"])

    # Определяем последний месяц по данным транзакций/переводов
    y, m = month_range_last(pd.concat([trans_all["date"], transfers_all["date"]], ignore_index=True))
    month_name = MONTHS_RU_GEN.get(m, "последнем месяце")

    trans_m = filter_month(trans_all, y, m, "date")
    transfers_m = filter_month(transfers_all, y, m, "date")

    month_feat = summarize_month(trans_m, transfers_m)

    # 3 месяца — для поведения (топапы/FX/снятия)
    has_topups = False
    fx_active = False
    low_withdrawals = True
    if not transfers_all.empty:
        topups_cnt = int(transfers_all.loc[transfers_all["type"].isin(["deposit_topup_out", "deposit_fx_topup_out"])].shape[0])
        has_topups = topups_cnt >= 2
        fx_cnt = int(transfers_all.loc[transfers_all["type"].isin(["fx_buy", "fx_sell"])].shape[0])
        fx_active = fx_cnt >= 2 or (not trans_all.empty and (trans_all["currency"].ne("KZT")).sum() >= 3)
        # низкие снятия — меньше 10% от среднего остатка
        total_atm = float(transfers_all.loc[transfers_all["type"] == "atm_withdrawal", "amount"].sum())
        low_withdrawals = total_atm < 0.1 * max(1.0, avg_balance)

    fx_curr = detect_fx_currency(trans_m, trans_all, transfers_m)

    # Выгоды по продуктам
    spend_total = month_feat["spend_total"]
    spend_map = month_feat["spend_by_cat"]

    benefit_map = {}

    # Карта для путешествий
    b_travel = benefit_travel(spend_map) if spend_map is not None else 0.0
    if b_travel > 500.0:  # минимальный порог пользы
        benefit_map["Карта для путешествий"] = b_travel

    # Премиальная карта
    b_prem = benefit_premium(spend_total, spend_map, avg_balance, month_feat["atm_withdrawal"], month_feat["card_p2p"])
    # показываем только если есть смысл: крупный остаток или заметная выгода
    if avg_balance >= 500_000 or b_prem >= 3000.0:
        benefit_map["Премиальная карта"] = b_prem

    # Кредитная карта
    b_credit = benefit_credit(spend_map) if spend_map is not None else 0.0
    if b_credit >= 2000.0:
        benefit_map["Кредитная карта"] = b_credit

    # FX/мультивалютный продукт (обмен валют)
    b_fx = benefit_fx(trans_m, transfers_m)
    if b_fx >= 500.0:
        benefit_map["Обмен валют"] = b_fx

    # Вклады — считаем для выбранного типа
    if avg_balance >= 200_000:
        dep_type = choose_deposit_type(avg_balance, fx_active, has_topups, low_withdrawals)
        b_dep = benefit_deposit(dep_type, avg_balance)
        # Немного приоритизируем сберегательный при очень высокой сумме
        benefit_map[dep_type] = b_dep

    # Кредит наличными — осторожно
    need_cash = detect_need_cash_loan(avg_balance, month_feat["inflows"], spend_total + month_feat["card_p2p"] + month_feat["utilities_out"] + month_feat["atm_withdrawal"])
    if need_cash:
        benefit_map["Кредит наличными"] = 1.0  # не считаем «выгоду» в деньгах, просто доп.слот

    # Инвестиции — запасной вариант (низкий приоритет)
    if avg_balance >= 50_000 and avg_balance < 300_000 and len(benefit_map) == 0:
        benefit_map["Инвестиции"] = 1.0

    # Если ничего не набралось — выберем наибольшее из того, что есть
    if len(benefit_map) == 0:
        # Слабый fallback: кредитная карта или инвестиции
        if b_credit > 0:
            benefit_map["Кредитная карта"] = b_credit
        else:
            benefit_map["Инвестиции"] = 1.0

    # Сортировка Top-N
    top_products = sorted(benefit_map.items(), key=lambda kv: kv[1], reverse=True)
    best_product, best_benefit = top_products[0]

    # Генерация пуша
    push = make_push(best_product, name, month_name, avg_balance, fx_curr, month_feat, spend_map, benefit_map[best_product])

    return {
        "client_code": client_code,
        "product": best_product,
        "push_notification": push,
        "debug": {
            "top4": top_products[:4],
            "month": f"{month_name}",
        }
    }

def make_push(product: str, name: str, month_name: str, avg_balance: float, fx_curr: str | None,
              mf: dict, spend_map: pd.Series, benefit_value: float) -> str:
    # Укорачиваем имя (без фамилии), обращение на "вы" по ТЗ
    name_short = name.split()[0] if name else "Вы"
    ben = fmt_kzt(benefit_value, 0)

    if product == "Карта для путешествий":
        taxi_cnt = mf["taxi_cnt"]
        taxi_sum = fmt_kzt(mf["taxi_sum"], 0)
        text = f"{name_short}, в {month_name} у вас {taxi_cnt} поездок на такси на {taxi_sum}. С картой для путешествий вернулась бы ≈{ben}. Открыть карту."
        return trim_to_push(text)

    if product == "Премиальная карта":
        # Оценим базовый % по остатку
        if avg_balance >= 6_000_000:
            base_pct = "4%"
        elif avg_balance >= 1_000_000:
            base_pct = "3%"
        else:
            base_pct = "2%"
        rest_sum = fmt_kzt(mf.get("restaurants_sum", 0.0), 0)
        bal = fmt_kzt(avg_balance, 0)
        text = f"{name_short}, у вас крупный остаток {bal} и траты в ресторанах на {rest_sum}. Премиальная карта даст до {base_pct} кешбэка и бесплатные снятия. Оформить сейчас."
        return trim_to_push(text)

    if product == "Кредитная карта":
        cats = top_categories(spend_map, 3)
        cats = [c for c in cats if c] or ["ежедневные траты"]
        cats_str = ", ".join(cats[:3])
        text = f"{name_short}, ваши топ-категории — {cats_str}. Кредитная карта даст до 10% в любимых и на онлайн‑сервисы, выгода ≈{ben}/мес. Оформить карту."
        return trim_to_push(text)

    if product == "Обмен валют":
        curr = fx_curr or "USD"
        text = f"{name_short}, вы часто платите в {curr}. В приложении выгодный обмен без комиссии и авто‑покупка по целевому курсу, экономия ≈{ben}/мес. Настроить обмен."
        return trim_to_push(text)

    if product in {"Депозит Сберегательный", "Депозит Накопительный", "Депозит Мультивалютный"}:
        rate = int(round(deposit_rate(product) * 100))
        bal = fmt_kzt(avg_balance, 0)
        text = f"{name_short}, у вас остаются свободные {bal}. Разместите на «{product}» — {rate}% годовых, доход ≈{ben}/мес. Открыть вклад."
        return trim_to_push(text)

    if product == "Кредит наличными":
        text = f"{name_short}, если нужен запас на крупные траты — оформите кредит наличными с гибкими выплатами. Узнать доступный лимит."
        return trim_to_push(text)

    if product == "Инвестиции":
        text = f"{name_short}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. Открыть счёт."
        return trim_to_push(text)

    # Fallback
    text = f"{name_short}, у нас есть продукт с выгодой ≈{ben} для ваших трат. Посмотреть предложение."
    return trim_to_push(text)

def trim_to_push(text: str, min_len=160, max_len=220) -> str:
    # Стримим к нужной длине: укорачиваем мягко, убираем двойные пробелы
    t = " ".join(text.split())
    if len(t) <= max_len:
        return t
    # если длиннее — урезаем в конце до ближайшей точки или полностью
    t_cut = t[:max_len]
    if "." in t_cut[-30:]:
        last_dot = t_cut.rfind(".")
        if last_dot > 0:
            return t_cut[:last_dot+1]
    return t_cut.rstrip(" ,;:-")

def main():
    clients = safe_read_csv(DATA_DIR / "clients.csv")
    if clients.empty:
        print("Не найден ./data/clients.csv")
        return

    out_rows = []
    debug_rows = []

    for _, row in clients.iterrows():
        try:
            res = analyze_client(row)
            out_rows.append({
                "client_code": res["client_code"],
                "product": res["product"],
                "push_notification": res["push_notification"]
            })
            debug_rows.append({
                "client_code": res["client_code"],
                "top4": json.dumps([(p, round(v)) for p, v in res["debug"]["top4"]], ensure_ascii=False),
                "month": res["debug"]["month"]
            })
        except Exception as e:
            cc = int(row["client_code"])
            out_rows.append({
                "client_code": cc,
                "product": "Инвестиции",
                "push_notification": f"{row['name']}, откройте инвестиционный счёт с нулевой комиссией на старт. Открыть счёт."
            })
            debug_rows.append({"client_code": cc, "top4": "[]", "month": ""})
            print(f"Клиент {cc}: ошибка {e}")

    out_df = pd.DataFrame(out_rows)
    out_path = OUT_DIR / "submissions_v2.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    dbg_df = pd.DataFrame(debug_rows)
    #dbg_df.to_csv(OUT_DIR / "debug_v2_top4.csv", index=False, encoding="utf-8-sig")

    print(f"Готово: {out_path} (и debug_v2_top4.csv)")

if __name__ == "__main__":
    main()