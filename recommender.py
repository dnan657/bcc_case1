# recommender.py
import numpy as np

def recommend_product(client_features):
    """
    ФИНАЛЬНАЯ ВЕРСИЯ "АРХЕТИП".
    Определяем доминирующий финансовый архетип клиента и предлагаем лучший продукт для него.
    Это гарантирует разнообразие и уместность.
    """
    status = client_features.get('status', 'Стандартный клиент')
    
    # --- ШАГ 1: Приоритетные сегменты (VIP и Студенты) ---
    # Эти сегменты настолько важны и очевидны, что мы обрабатываем их в первую очередь.

    avg_balance = client_features.get('avg_monthly_balance_KZT', 0)
    if avg_balance > 1_500_000 or status == 'Премиальный клиент':
        return 'Премиальная карта'
        
    # Потенциал к накоплению (оцениваем по среднему остатку и зарплате)
    potential_deposit = max(avg_balance, client_features.get('salary_3m', 0) / 3 * 0.25)
    
    if status == 'Студент' and potential_deposit > 50_000:
        return 'Инвестиции'

    # --- ШАГ 2: Определяем доминирующий архетип для всех остальных ---
    
    archetype_scores = {
        'Сберегатель': potential_deposit,
        # Умножаем на весовой коэффициент, чтобы траты на поездки были сопоставимы с накоплениями
        'Путешественник': client_features.get('travel_spend_3m', 0) * 5,
        'Тратящий': client_features.get('discretionary_spend_3m', 0)
    }

    # --- ШАГ 3: Выносим вердикт на основе доминирующего архетипа ---

    # Находим архетип с максимальной оценкой
    if not any(archetype_scores.values()): # Если все нули
         return 'Кредитная карта' # Безопасный вариант по умолчанию
         
    dominant_archetype = max(archetype_scores, key=archetype_scores.get)

    if dominant_archetype == 'Сберегатель':
        # Если клиент - сберегатель, предлагаем ему лучший вклад
        if client_features.get('fx_operations_count_3m', 0) > 2:
            return 'Депозит Мультивалютный'
        else:
            return 'Депозит Сберегательный' # Самый выгодный по ставке

    elif dominant_archetype == 'Путешественник':
        return 'Карта для путешествий'
        
    elif dominant_archetype == 'Тратящий':
        return 'Кредитная карта'

    return 'Кредитная карта' # Запасной вариант