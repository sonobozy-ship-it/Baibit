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

# Сброс кривого paper state (баланс без учёта маржи).
# Удаляем только если версия стейта не содержит поля margin (старый формат).
PAPER_STATE="$APP_DIR/bybit-bot-v5/backend/data/paper_state.json"
if [ -f "$PAPER_STATE" ]; then
    if ! python3 -c "
import json, sys
d = json.load(open('$PAPER_STATE'))
positions = d.get('positions', {})
# Если есть хоть одна позиция без поля margin — стейт старый, сбрасываем
if any('margin' not in p for p in positions.values()):
    sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
        INIT_BAL=$(grep -oP '(?<=PAPER_INITIAL_BALANCE=)\S+' "$ENV_FILE" 2>/dev/null || echo "200")
        python3 -c "
import json
d = {'balance': $INIT_BAL, 'positions': {}, 'trades_history': [], 'saved_at': ''}
open('$PAPER_STATE', 'w').write(json.dumps(d, indent=2))
"
        echo "   ⚠️  Paper state сброшен (старый формат без маржи). Баланс: ${INIT_BAL} USDT"
    fi
fi

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
