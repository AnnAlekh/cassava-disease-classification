# 🚀 Быстрый старт - Публикация на GitHub

## Быстрая инструкция

### 1. Создайте репозиторий на GitHub
- Перейдите на https://github.com/new
- Имя: `cassava-disease-classification` (или другое)
- Тип: **Public**
- **НЕ** ставьте галочки на README, .gitignore, license

### 2. Подключите локальный репозиторий к GitHub

```bash
# Замените YOUR_USERNAME на ваш GitHub username
git remote add origin https://github.com/YOUR_USERNAME/cassava-disease-classification.git
```

### 3. Отправьте все ветки на GitHub

```bash
# Отправить все ветки
git push --all origin

# Если нужна аутентификация, используйте Personal Access Token вместо пароля
```

## Структура веток

- **main** - продакшен версия
- **develop** - разработка
- **feature/model-improvements** - улучшения модели
- **feature/api-enhancements** - улучшения API
- **feature/mobile-offline** - офлайн режим
- **bugfix/fixes** - исправления

## Полная инструкция

См. файл `SETUP_GITHUB.md` для детальной инструкции.
