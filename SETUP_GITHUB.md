# 📤 Инструкция по публикации репозитория на GitHub

## Шаг 1: Создание репозитория на GitHub

### Через веб-интерфейс GitHub:

1. Перейдите на [GitHub](https://github.com) и войдите в свой аккаунт
2. Нажмите кнопку **"+"** в правом верхнем углу → **"New repository"**
3. Заполните форму:
   - **Repository name**: `cassava-disease-classification` (или другое имя на ваш выбор)
   - **Description**: `Система для автоматической классификации болезней листьев кассавы с использованием глубокого обучения`
   - **Visibility**: Выберите **Public** (публичный репозиторий)
   - **НЕ** устанавливайте галочки на:
     - ❌ Add a README file (у нас уже есть README.md)
     - ❌ Add .gitignore (у нас уже есть .gitignore)
     - ❌ Choose a license (можно добавить позже)
4. Нажмите кнопку **"Create repository"**

### Через GitHub CLI (если установлен):

```bash
gh repo create cassava-disease-classification --public --description "Система для автоматической классификации болезней листьев кассавы" --source=. --remote=origin --push
```

## Шаг 2: Добавление удаленного репозитория

После создания репозитория на GitHub, скопируйте URL репозитория (он будет выглядеть так: `https://github.com/ваш-username/cassava-disease-classification.git`)

Затем выполните команды:

```bash
# Добавьте удаленный репозиторий
git remote add origin https://github.com/ваш-username/cassava-disease-classification.git

# Проверьте, что удаленный репозиторий добавлен
git remote -v
```

## Шаг 3: Отправка всех веток на GitHub

```bash
# Убедитесь, что вы на ветке main
git checkout main

# Отправьте ветку main на GitHub
git push -u origin main

# Отправьте ветку develop
git checkout develop
git push -u origin develop

# Отправьте все остальные ветки
git push origin feature/model-improvements
git push origin feature/api-enhancements
git push origin feature/mobile-offline
git push origin bugfix/fixes
```

### Альтернатива: Отправить все ветки одной командой

```bash
# Отправить все ветки одновременно
git push --all origin

# Отправить все теги (если есть)
git push --tags origin
```

## Шаг 4: Настройка ветки по умолчанию (опционально)

На GitHub по умолчанию может быть установлена ветка `master`. Если хотите, чтобы по умолчанию была ветка `main`:

1. Перейдите в настройки репозитория на GitHub: **Settings** → **Branches**
2. В разделе **Default branch** выберите `main`
3. Нажмите **Update**

## Шаг 5: Проверка

После выполнения команд:

1. Обновите страницу репозитория на GitHub
2. Убедитесь, что все файлы загружены
3. Проверьте, что README.md отображается корректно
4. Убедитесь, что все ветки видны на странице **Branches**

## Структура веток в репозитории

### Основные ветки:
- **main** - Стабильная версия для продакшена
- **develop** - Основная ветка разработки

### Ветки для разработки:
- **feature/model-improvements** - Улучшения модели машинного обучения
- **feature/api-enhancements** - Улучшения API
- **feature/mobile-offline** - Офлайн режим для мобильного приложения

### Ветки для исправлений:
- **bugfix/fixes** - Исправления багов

## Последующие коммиты и пуш

После внесения изменений в проект:

```bash
# Добавить изменения
git add .

# Создать коммит
git commit -m "Описание изменений"

# Отправить в текущую ветку
git push

# Или указать ветку явно
git push origin имя-ветки
```

## Настройка GitHub Actions (опционально)

Для автоматизации CI/CD можно добавить `.github/workflows/ci.yml`. Пример:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

## Добавление лицензии (опционально)

Если хотите добавить лицензию MIT:

```bash
# Создать файл LICENSE
cat > LICENSE << EOF
MIT License

Copyright (c) 2024 [Ваше имя]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Добавить в git
git add LICENSE
git commit -m "Add MIT License"
git push
```

## Полезные команды

```bash
# Посмотреть все ветки
git branch -a

# Посмотреть удаленные репозитории
git remote -v

# Обновить информацию о ветках на GitHub
git fetch origin

# Синхронизировать локальную ветку с удаленной
git pull origin main

# Клонировать репозиторий (для другого компьютера)
git clone https://github.com/ваш-username/cassava-disease-classification.git
```

## Решение проблем

### Если получили ошибку "remote origin already exists":
```bash
# Удалить существующий remote
git remote remove origin

# Добавить снова
git remote add origin https://github.com/ваш-username/cassava-disease-classification.git
```

### Если нужно изменить URL удаленного репозитория:
```bash
git remote set-url origin https://github.com/новый-username/новое-имя-репозитория.git
```

### Если забыли имя пользователя или токен:
Настройте GitHub CLI или используйте Personal Access Token:
1. GitHub → Settings → Developer settings → Personal access tokens → Generate new token
2. Используйте токен вместо пароля при push

---

🎉 После выполнения всех шагов ваш проект будет публично доступен на GitHub!

