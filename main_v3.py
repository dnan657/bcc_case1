from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# ────────────────────────────────
#  CONFIG & CONSTANTS
# ────────────────────────────────
DATA_DIR = Path("./data")
DEFAULT_OUT = Path("./submissions_v3.csv")
MAX_PUSH_LEN = 220
MIN_PUSH_LEN = 160

TRAVEL_CATS = {"Такси", "Путешествия", "Отели"}
ONLINE_CATS = {"Играем дома", "Смотрим дома", "Кино", "Едим дома"}
JPR_CATS = {"Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"}

MONTHS_RU_GEN = {
    1: "январе", 2: "феврале", 3: "марте", 4: "апреле",
    5: "мае", 6: "июне", 7: "июле", 8: "августе",
    9: "сентябре", 10: "октябре", 11: "ноябре", 12: "декабре",
}

pd.options.mode.chained_assignment = None  # noqa: E402

# ────────────────────────────────
#  HELPERS
# ────────────────────────────────
def fmt_kzt(val: float, decimals: int = 0) -> str:
    if np.isnan(val):
        return "0 ₸"
    q = round(float(val), decimals)
    s = f"{q:,.{decimals}f}".replace(",", " ").replace(".", ",")
    return f"{s} ₸"


def trim_to_push(txt: str, *, max_len: int = MAX_PUSH_LEN) -> str:
    t = " ".join(txt.split())
    if len(t) <= max_len:
        return t
    cut = t[: max_len + 1]
    last_dot = cut.rfind(".")
    if 0 < last_dot < max_len - 10:
        return cut[: last_dot + 1]
    return cut.rstrip(" ,;:-")


def safe_read_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """
    Быстрый парсер CSV:
    1) пытаемся pyarrow-engine (самый быстрый);
    2) если не получилось → fallback на стандартный C-engine.
    """
    if not path.exists():
        return pd.DataFrame()

    common_kwargs = dict(encoding="utf-8-sig", parse_dates=parse_dates)

    try:
        return pd.read_csv(
            path,
            engine="pyarrow",          # быстрый вариант
            dtype_backend="pyarrow",
            **common_kwargs,
        )
    except (ValueError, ImportError):
        # pyarrow недоступен или не поддерживает какие-то опции
        return pd.read_csv(
            path,
            engine="c",
            low_memory=False,          # теперь можно
            **common_kwargs,
        )
    except UnicodeDecodeError:
        # редкий случай с другой кодировкой
        return pd.read_csv(
            path,
            engine="c",
            low_memory=False,
            encoding="utf-8",
            parse_dates=parse_dates,
        )


def month_of_last_date(dates: pd.Series) -> Tuple[int, int]:
    if dates.empty:
        now = datetime.now()
        return now.year, now.month
    last = pd.to_datetime(dates.max())
    return last.year, last.month


def filter_month(df: pd.DataFrame, *, year: int, month: int) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d[(d["date"].dt.year == year) & (d["date"].dt.month == month)]


def top_categories(sp_by_cat: pd.Series, n: int = 3) -> List[str]:
    if sp_by_cat.empty:
        return []
    return sp_by_cat.sort_values(ascending=False).head(n).index.tolist()


def union_spend(spend_map: pd.Series, cats: set[str]) -> float:
    if spend_map.empty:
        return 0.0
    return float(spend_map.reindex(cats, fill_value=0.0).sum())


def detect_fx_currency(
    trans_m: pd.DataFrame, trans_all: pd.DataFrame, trf_m: pd.DataFrame
) -> str | None:
    if not trans_m.empty:
        cur_counts = trans_m.loc[trans_m["currency"].ne("KZT"), "currency"].value_counts()
        if not cur_counts.empty:
            return str(cur_counts.idxmax())
    if not trf_m.empty:
        fx = trf_m[trf_m["type"].isin(["fx_buy", "fx_sell"])]
        if not fx.empty and not fx["currency"].isna().all():
            return str(fx["currency"].mode(dropna=True).iloc[0])
    return None


# ────────────────────────────────
#  DATA CLASSES
# ────────────────────────────────
@dataclass
class MonthMetrics:
    spend_total: float
    spend_by_cat: pd.Series
    taxi_cnt: int
    taxi_sum: float
    restaurants_sum: float
    online_sum: float
    atm_withdrawal: float
    card_p2p: float
    inflows: float
    utilities_out: float


@dataclass
class ClientResult:
    client_code: int
    product: str
    push_notification: str
    top4: List[Tuple[str, float]]
    month_ru: str


# ────────────────────────────────
#  BENEFIT CALCULATIONS
# ────────────────────────────────
def benefit_travel(spend_map: pd.Series) -> float:
    return 0.04 * union_spend(spend_map, TRAVEL_CATS)


def benefit_premium(
    total_spend: float,
    spend_map: pd.Series,
    avg_balance: float,
    atm_withdrawal: float,
    card_p2p: float,
) -> float:
    base = 0.04 if avg_balance >= 6_000_000 else 0.03 if avg_balance >= 1_000_000 else 0.02
    cashback = base * total_spend
    extra = max(0.0, 0.04 - base) * union_spend(spend_map, JPR_CATS)
    cashback = min(cashback + extra, 100_000.0)
    atm_save = min(atm_withdrawal, 3_000_000.0) * 0.003
    trf_save = card_p2p * 0.002
    return cashback + atm_save + trf_save


def benefit_credit(spend_map: pd.Series) -> float:
    cats = set(top_categories(spend_map, 3)) | ONLINE_CATS
    return 0.10 * union_spend(spend_map, cats)


def benefit_fx(trans_m: pd.DataFrame, trf_m: pd.DataFrame) -> float:
    fx_trf = trf_m.loc[trf_m["type"].isin(["fx_buy", "fx_sell"]), "amount"].sum() if not trf_m.empty else 0
    non_kzt = trans_m.loc[trans_m["currency"].ne("KZT"), "amount"].sum() if not trans_m.empty else 0
    return 0.005 * float(fx_trf + non_kzt)


def choose_deposit_type(
    avg_balance: float, fx_active: bool, has_topups: bool, low_withdrawals: bool
) -> str:
    if fx_active:
        return "Депозит Мультивалютный"
    if has_topups:
        return "Депозит Накопительный"
    if low_withdrawals:
        return "Депозит Сберегательный"
    return "Депозит Накопительный"


def deposit_rate(product: str) -> float:
    return {"Депозит Сберегательный": 0.165, "Депозит Накопительный": 0.155, "Депозит Мультивалютный": 0.145}.get(
        product, 0.0
    )


def benefit_deposit(product: str, avg_balance: float) -> float:
    return (deposit_rate(product) / 12) * avg_balance


def need_cash_loan(avg_balance: float, inflows: float, outflows: float) -> bool:
    return (outflows - inflows) > 100_000 and avg_balance < 150_000


# ────────────────────────────────
#  PUSH GENERATOR
# ────────────────────────────────
def make_push(
    product: str,
    name: str,
    month_ru: str,
    avg_balance: float,
    fx_curr: str | None,
    m: MonthMetrics,
    spend_map: pd.Series,
    benefit_val: float,
) -> str:
    user = name.split()[0] if name else "Вы"
    ben = fmt_kzt(benefit_val, 0)

    if product == "Карта для путешествий":
        tx_sum = fmt_kzt(m.taxi_sum, 0)
        text = f"{user}, в {month_ru} у вас {m.taxi_cnt} поездок на такси на {tx_sum}. С тревел-картой вернулось бы ≈{ben}. Открыть карту."
        return trim_to_push(text)

    if product == "Премиальная карта":
        base_pct = "4%" if avg_balance >= 6_000_000 else "3%" if avg_balance >= 1_000_000 else "2%"
        rest_sum = fmt_kzt(m.restaurants_sum, 0)
        bal = fmt_kzt(avg_balance, 0)
        text = (
            f"{user}, остаток {bal} и рестораны на {rest_sum}. Премиальная карта даёт до {base_pct} кешбэка и бесплатные снятия. Оформить."
        )
        return trim_to_push(text)

    if product == "Кредитная карта":
        cats = ", ".join(top_categories(spend_map, 3) or ["ежедневные траты"])
        text = f"{user}, ваши топ-категории — {cats}. Кредитная карта даст до 10% и рассрочку, выгода ≈{ben}/мес. Оформить."
        return trim_to_push(text)

    if product == "Обмен валют":
        curr = fx_curr or "USD"
        text = (
            f"{user}, вы часто платите в {curr}. В приложении обмен без комиссии и авто-покупка по курсу, экономия ≈{ben}/мес. Настроить."
        )
        return trim_to_push(text)

    if product.startswith("Депозит"):
        rate = int(round(deposit_rate(product) * 100))
        bal = fmt_kzt(avg_balance, 0)
        text = f"{user}, свободно {bal}. «{product}» — {rate}% годовых, доход ≈{ben}/мес. Открыть вклад."
        return trim_to_push(text)

    if product == "Кредит наличными":
        return trim_to_push(
            f"{user}, нужен запас на крупные траты? Кредит наличными с гибкими выплатами доступен онлайн. Узнать лимит."
        )

    if product == "Инвестиции":
        return trim_to_push(
            f"{user}, попробуйте инвестиции с нулевой комиссией на старт и порогом от 6 ₸. Открыть счёт."
        )

    return trim_to_push(f"{user}, у нас есть предложение с выгодой ≈{ben}. Посмотреть.")


# ────────────────────────────────
#  CORE PIPELINE
# ────────────────────────────────
def summarize_month(trans_m: pd.DataFrame, trf_m: pd.DataFrame) -> MonthMetrics:
    spend_total = float(trans_m["amount"].sum()) if not trans_m.empty else 0.0
    spend_by_cat = (
        trans_m.groupby("category")["amount"].sum() if not trans_m.empty else pd.Series(dtype=float)
    )
    taxi_mask = trans_m["category"] == "Такси" if not trans_m.empty else []
    rest_mask = trans_m["category"] == "Кафе и рестораны" if not trans_m.empty else []

    return MonthMetrics(
        spend_total=spend_total,
        spend_by_cat=spend_by_cat,
        taxi_cnt=int(taxi_mask.sum()) if not trans_m.empty else 0,
        taxi_sum=float(trans_m.loc[taxi_mask, "amount"].sum()) if not trans_m.empty else 0.0,
        restaurants_sum=float(trans_m.loc[rest_mask, "amount"].sum()) if not trans_m.empty else 0.0,
        online_sum=union_spend(spend_by_cat, ONLINE_CATS),
        atm_withdrawal=float(trf_m.loc[trf_m["type"] == "atm_withdrawal", "amount"].sum())
        if not trf_m.empty
        else 0.0,
        card_p2p=float(
            trf_m.loc[trf_m["type"].isin(["card_out", "p2p_out"]), "amount"].sum()
        )
        if not trf_m.empty
        else 0.0,
        inflows=float(trf_m.loc[trf_m["direction"] == "in", "amount"].sum())
        if not trf_m.empty
        else 0.0,
        utilities_out=float(trf_m.loc[trf_m["type"] == "utilities_out", "amount"].sum())
        if not trf_m.empty
        else 0.0,
    )


def analyze_client(row: pd.Series) -> ClientResult:
    code = int(row["client_code"])
    name = str(row["name"])
    avg_balance = float(row["avg_monthly_balance_KZT"])

    tr_file = DATA_DIR / f"client_{code}_transactions_3m.csv"
    tf_file = DATA_DIR / f"client_{code}_transfers_3m.csv"

    trans_all = safe_read_csv(tr_file, parse_dates=["date"])
    transfers_all = safe_read_csv(tf_file, parse_dates=["date"])

    y, m = month_of_last_date(pd.concat([trans_all["date"], transfers_all["date"]], ignore_index=True))
    month_ru = MONTHS_RU_GEN.get(m, "последнем месяце")

    trans_m = filter_month(trans_all, year=y, month=m)
    trf_m = filter_month(transfers_all, year=y, month=m)

    mm = summarize_month(trans_m, trf_m)

    has_topups = transfers_all["type"].isin(["deposit_topup_out", "deposit_fx_topup_out"]).sum() >= 2
    fx_active = (
        transfers_all["type"].isin(["fx_buy", "fx_sell"]).sum() >= 2
        or trans_all["currency"].ne("KZT").sum() >= 3
    )
    low_withdrawals = (
        transfers_all.loc[transfers_all["type"] == "atm_withdrawal", "amount"].sum()
        < 0.1 * max(1.0, avg_balance)
    )
    fx_curr = detect_fx_currency(trans_m, trans_all, trf_m)

    # ── Benefits
    benefits: Dict[str, float] = {}
    if (bt := benefit_travel(mm.spend_by_cat)) > 500:
        benefits["Карта для путешествий"] = bt

    bp = benefit_premium(
        mm.spend_total, mm.spend_by_cat, avg_balance, mm.atm_withdrawal, mm.card_p2p
    )
    if avg_balance >= 500_000 or bp >= 3_000:
        benefits["Премиальная карта"] = bp

    bc = benefit_credit(mm.spend_by_cat)
    if bc >= 2_000:
        benefits["Кредитная карта"] = bc

    bf = benefit_fx(trans_m, trf_m)
    if bf >= 500:
        benefits["Обмен валют"] = bf

    if avg_balance >= 200_000:
        dep_type = choose_deposit_type(avg_balance, fx_active, has_topups, low_withdrawals)
        benefits[dep_type] = benefit_deposit(dep_type, avg_balance)

    outflows_all = (
        mm.spend_total + mm.card_p2p + mm.utilities_out + mm.atm_withdrawal
    )
    if need_cash_loan(avg_balance, mm.inflows, outflows_all):
        benefits["Кредит наличными"] = 1.0

    if not benefits and 50_000 <= avg_balance < 300_000:
        benefits["Инвестиции"] = 1.0

    if not benefits:
        benefits["Инвестиции"] = 1.0  # крайний fallback

    top4 = sorted(benefits.items(), key=lambda kv: kv[1], reverse=True)[:4]
    best_product, best_benefit = top4[0]

    push = make_push(
        best_product, name, month_ru, avg_balance, fx_curr, mm, mm.spend_by_cat, best_benefit
    )
    return ClientResult(code, best_product, push, top4, month_ru)


# ────────────────────────────────
#  MAIN
# ────────────────────────────────
def run(output_path: Path, debug: bool = False, workers: int = 4) -> None:
    clients = safe_read_csv(DATA_DIR / "clients.csv")
    if clients.empty:
        raise FileNotFoundError("Не найден clients.csv")

    results: List[ClientResult] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(analyze_client, row): idx for idx, row in clients.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing clients"):
            try:
                results.append(fut.result())
            except Exception as e:
                idx = futures[fut]
                code = int(clients.iloc[idx]["client_code"])
                logging.exception("Client %s failed: %s", code, e)

    sub_df = pd.DataFrame(
        [
            {"client_code": r.client_code, "product": r.product, "push_notification": r.push_notification}
            for r in results
        ]
    )
    sub_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logging.info("Saved %s", output_path)

    if debug:
        dbg = pd.DataFrame(
            [
                {
                    "client_code": r.client_code,
                    "month": r.month_ru,
                    "top4": json.dumps([(p, round(v, 1)) for p, v in r.top4], ensure_ascii=False),
                }
                for r in results
            ]
        )
        dbg.to_csv(output_path.with_stem(output_path.stem + "_debug"), index=False, encoding="utf-8-sig")
        logging.info("Saved debug")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate personalised pushes CSV.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--debug", action="store_true", help="Dump top-4 diagnostics")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (threads)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(args.output, debug=args.debug, workers=args.workers)