# feature_generator.py
import pandas as pd

# Определяем константы для категорий, чтобы код был чище
TRAVEL_CATS = ['Такси', 'Отели', 'Путешествия']
PREMIUM_CATS = ['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны', 'Спа и массаж']
ONLINE_CATS = ['Смотрим дома', 'Играем дома', 'Кино']
# Категории трат "для себя", не являющиеся обязательными
DISCRETIONARY_CATS = PREMIUM_CATS + ONLINE_CATS + ['Одежда и обувь', 'Развлечения', 'Подарки', 'Хобби']

def generate_features(client_profile, transactions, transfers):
    """
    Агрегирует сырые данные в осмысленные признаки (фичи) для одного клиента.
    """
    features = client_profile.to_dict()

    # 1. Анализ трат (транзакции)
    if not transactions.empty:
        features['total_spend_3m'] = transactions['amount'].sum()
        features['travel_spend_3m'] = transactions[transactions['category'].isin(TRAVEL_CATS)]['amount'].sum()
        features['premium_spend_3m'] = transactions[transactions['category'].isin(PREMIUM_CATS)]['amount'].sum()
        features['online_spend_3m'] = transactions[transactions['category'].isin(ONLINE_CATS)]['amount'].sum()
        features['discretionary_spend_3m'] = transactions[transactions['category'].isin(DISCRETIONARY_CATS)]['amount'].sum()

        top_cats = transactions.groupby('category')['amount'].sum().nlargest(3)
        features['top_1_category'] = top_cats.index[0] if len(top_cats) > 0 else ''
        features['top_2_category'] = top_cats.index[1] if len(top_cats) > 1 else ''
        features['top_3_category'] = top_cats.index[2] if len(top_cats) > 2 else ''
    else:
        # Если транзакций нет, ставим нули
        for key in ['total_spend_3m', 'travel_spend_3m', 'premium_spend_3m', 'online_spend_3m', 'discretionary_spend_3m']:
            features[key] = 0
        for key in ['top_1_category', 'top_2_category', 'top_3_category']:
            features[key] = ''

    # 2. Анализ переводов и накоплений
    if not transfers.empty:
        fx_ops = transfers[transfers['type'].isin(['fx_buy', 'fx_sell'])]
        features['fx_operations_count_3m'] = len(fx_ops)
        features['fx_volume_3m'] = fx_ops['amount'].sum()
        
        salary_in = transfers[transfers['type'] == 'salary_in']
        features['salary_3m'] = salary_in['amount'].sum()
        
        invest_ops = transfers[transfers['type'].isin(['invest_in', 'invest_out'])]
        features['has_investments'] = not invest_ops.empty
    else:
        # Если переводов нет
        features['fx_operations_count_3m'] = 0
        features['fx_volume_3m'] = 0
        features['salary_3m'] = 0
        features['has_investments'] = False

    return features