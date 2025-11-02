# Cassava Disease Detector - Flutter MVP

Мобильное приложение для диагностики болезней листьев кассавы на Flutter.

## Возможности

- 📸 **Сканирование листьев**: Фото через камеру или галерею
- 🔍 **Мгновенная диагностика**: Определение 5 типов болезней + здоровые растения
- 💡 **Рекомендации по лечению**: Полезные советы для каждого диагноза
- 📊 **История сканирований**: Локальное хранение всех результатов
- 🎨 **Современный UI**: Красивый и интуитивный интерфейс

## Требования

- Flutter SDK >= 3.0.0
- Dart SDK >= 3.0.0
- Запущенный API сервер (FastAPI на порту 8000)

## Установка

1. Установите зависимости:
```bash
cd mobile_app
flutter pub get
```

2. Убедитесь, что API сервер запущен:
```bash
cd ..
docker-compose up -d
# или
python app.py
```

3. Настройте URL API (если нужно):
   - Для Android эмулятора: используется `http://10.0.2.2:8000` по умолчанию
   - Для iOS симулятора или реального устройства: замените на IP вашего компьютера в `lib/services/api_service.dart`

## Запуск

### Android
```bash
flutter run
```

### iOS
```bash
flutter run
```

## Структура проекта

```
mobile_app/
├── lib/
│   ├── main.dart                 # Точка входа приложения
│   ├── models/                   # Модели данных
│   │   ├── prediction_result.dart
│   │   └── scan_history.dart
│   ├── screens/                  # Экраны приложения
│   │   ├── home_screen.dart      # Главный экран
│   │   ├── camera_screen.dart    # Экран сканирования
│   │   ├── result_screen.dart    # Экран результатов
│   │   └── history_screen.dart   # История сканирований
│   └── services/                 # Сервисы
│       ├── api_service.dart      # API клиент
│       └── database_service.dart # Локальная БД
├── pubspec.yaml                  # Зависимости
└── README.md
```

## Настройка API URL

Если API запущен не на localhost, измените URL в `lib/services/api_service.dart`:

```dart
// Для реального устройства замените на IP вашего компьютера:
baseUrl = 'http://192.168.1.XXX:8000';
```

## Диагностируемые болезни

1. **Cassava Bacterial Blight (CBB)** - Бактериальный ожог
2. **Cassava Brown Streak Disease (CBSD)** - Болезнь коричневых полос
3. **Cassava Green Mottle (CGM)** - Зеленая крапчатость
4. **Cassava Mosaic Disease (CMD)** - Мозаичная болезнь
5. **Healthy** - Здоровое растение

## Лицензия

Это MVP проект для демонстрации возможностей.

