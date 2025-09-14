# main.py
import pandas as pd
import os
from tqdm import tqdm # Библиотека для красивого отображения прогресса

# Импортируем наши модули
from feature_generator import generate_features
from recommender import recommend_product
from push_generator import generate_push

def main():
    """
    Главный скрипт, который оркестрирует весь процесс:
    1. Загружает данные по клиентам.
    2. Итерируется по каждому клиенту.
    3. Генерирует фичи.
    4. Выбирает продукт.
    5. Создает пуш-уведомление.
    6. Сохраняет результат.
    """
    DATA_PATH = './data/'
    CLIENTS_FILE = os.path.join(DATA_PATH, 'clients.csv')
    
    try:
        clients_df = pd.read_csv(CLIENTS_FILE)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {CLIENTS_FILE} не найден. Убедитесь, что он лежит в папке 'data'.")
        return

    results = []
    
    print("Начинаю обработку клиентов...")
    # tqdm оборачивает итератор, чтобы показать progress bar
    for _, client_profile in tqdm(clients_df.iterrows(), total=clients_df.shape[0]):
        client_code = client_profile['client_code']
        
        # Загружаем транзакции и переводы для текущего клиента
        transactions_file = os.path.join(DATA_PATH, f"client_{client_code}_transactions_3m.csv")
        transfers_file = os.path.join(DATA_PATH, f"client_{client_code}_transfers_3m.csv")
        
        try:
            transactions = pd.read_csv(transactions_file)
            transfers = pd.read_csv(transfers_file)
        except FileNotFoundError:
            # Если данных по клиенту нет, пропускаем его
            continue
            
        # Шаг 1: Генерируем фичи (поведенческие метрики)
        client_features = generate_features(client_profile, transactions, transfers)
        
        # Шаг 2: Рекомендуем продукт на основе фичей
        recommended_product = recommend_product(client_features)
        
        # Шаг 3: Генерируем текст пуш-уведомления
        push_notification = generate_push(client_features, recommended_product)
        
        results.append({
            'client_code': client_code,
            'product': recommended_product,
            'push_notification': push_notification
        })

    # Сохраняем итоговый файл
    output_df = pd.DataFrame(results)
    output_df.to_csv('submission.csv', index=False)
    
    print("\nГотово! Результаты сохранены в файл submission.csv")
    print("Пример результата:")
    print(output_df.head())

if __name__ == "__main__":
    main()