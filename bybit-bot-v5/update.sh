#!/bin/bash
# Обновление бота без затирания ключей и БД
set -e

APP_DIR="/opt/baibit"
BRANCH="claude/analyze-repository-files-TvlQi"
VENV="$APP_DIR/bybit-bot-v5/venv/bin/activate"

echo ""
echo "🔄 Обновление Baibit..."

cd "$APP_DIR"

# .env и data/ в .gitignore — git их не трогает.
# Для надёжности делаем резервную копию .env перед любыми git-операциями.
ENV_FILE="$APP_DIR/bybit-bot-v5/backend/.env"
ENV_BACKUP="/tmp/baibit_env_backup"
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_BACKUP"
    echo "   .env сохранён во временный бэкап"
fi

# Стягиваем новый код
git fetch origin "$BRANCH" -q
git reset --hard "origin/$BRANCH" -q
echo "   Код обновлён"

# Восстанавливаем .env если git его случайно затёр
if [ -f "$ENV_BACKUP" ] && [ ! -f "$ENV_FILE" ]; then
    cp "$ENV_BACKUP" "$ENV_FILE"
    echo "   .env восстановлен из бэкапа"
fi

# Обновляем Python-зависимости (новые пакеты из requirements.txt)
if [ -f "$VENV" ]; then
    echo "   Обновление зависимостей..."
    source "$VENV"
    pip install -r "$APP_DIR/bybit-bot-v5/requirements.txt" -q
    echo "   Зависимости актуальны"
fi

# Создаём директории если их нет (git не хранит пустые папки)
mkdir -p "$APP_DIR/bybit-bot-v5/backend/data/candles"
mkdir -p "$APP_DIR/bybit-bot-v5/backend/data/features"
mkdir -p "$APP_DIR/bybit-bot-v5/backend/data/models"
mkdir -p "$APP_DIR/bybit-bot-v5/backend/logs"

systemctl restart baibit
sleep 3

if systemctl is-active --quiet baibit; then
    echo "✅ Готово! Бот обновлён и запущен."
    echo "   Откройте Telegram и напишите /menu"
else
    echo "❌ Ошибка запуска. Логи:"
    journalctl -u baibit -n 30 --no-pager
fi
echo ""
