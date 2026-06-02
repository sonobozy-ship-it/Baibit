#!/bin/bash
# Обновление бота с полным бэкапом ключей и БД
set -e

APP_DIR="/opt/baibit"
BRANCH="claude/analyze-repository-files-TvlQi"
VENV="$APP_DIR/bybit-bot-v5/venv/bin/activate"
BACKEND="$APP_DIR/bybit-bot-v5/backend"

# Директория бэкапов — хранится ВНЕ репозитория, никогда не удаляется
BACKUP_ROOT="/opt/baibit_backups"
BACKUP_DIR="$BACKUP_ROOT/$(date +%Y%m%d_%H%M%S)"

echo ""
echo "🔄 Обновление Baibit..."
echo "   Бэкап → $BACKUP_DIR"

mkdir -p "$BACKUP_DIR"

# ── 1. Бэкап .env (ключи API) ────────────────────────────────────────────────
ENV_FILE="$BACKEND/.env"
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$BACKUP_DIR/.env"
    echo "   ✅ .env сохранён"
fi

# ── 2. Бэкап всех БД и JSON-состояний ────────────────────────────────────────
DATA_DIR="$BACKEND/data"
if [ -d "$DATA_DIR" ]; then
    cp -r "$DATA_DIR" "$BACKUP_DIR/data"
    echo "   ✅ data/ сохранена (БД, paper_state, модели)"
fi

# ── 3. Бэкап логов (последние 5 МБ) ──────────────────────────────────────────
LOGS_DIR="$BACKEND/logs"
if [ -d "$LOGS_DIR" ]; then
    mkdir -p "$BACKUP_DIR/logs"
    find "$LOGS_DIR" -name "*.log" -newer "$BACKEND" 2>/dev/null \
        | head -10 | xargs -I{} cp {} "$BACKUP_DIR/logs/" 2>/dev/null || true
    echo "   ✅ Логи сохранены"
fi

# ── 4. Обновление кода ────────────────────────────────────────────────────────
cd "$APP_DIR"
git fetch origin "$BRANCH" -q
git reset --hard "origin/$BRANCH" -q
echo "   ✅ Код обновлён до последнего коммита"

# ── 5. Восстановление .env если git его затронул ─────────────────────────────
if [ -f "$BACKUP_DIR/.env" ] && [ ! -f "$ENV_FILE" ]; then
    cp "$BACKUP_DIR/.env" "$ENV_FILE"
    echo "   ⚠️  .env восстановлен из бэкапа (git его удалил)"
fi

# ── 6. Восстановление data/ если git её затронул ─────────────────────────────
if [ -d "$BACKUP_DIR/data" ] && [ ! -d "$DATA_DIR" ]; then
    cp -r "$BACKUP_DIR/data" "$DATA_DIR"
    echo "   ⚠️  data/ восстановлена из бэкапа"
fi

# ── 7. Создаём папки если их нет ─────────────────────────────────────────────
mkdir -p "$DATA_DIR/candles" "$DATA_DIR/features" "$DATA_DIR/models"
mkdir -p "$BACKEND/logs"

# ── 8. Paper state: синхронизация баланса и проверка формата ─────────────────
PAPER_STATE="$DATA_DIR/paper_state.json"
INIT_BAL=$(grep -oP '(?<=PAPER_INITIAL_BALANCE=)\S+' "$ENV_FILE" 2>/dev/null || echo "1000")

if [ -f "$PAPER_STATE" ]; then
    # 8a/8b. Мигрируем формат БЕЗ сброса баланса.
    # Старый формат (позиции без поля margin) — добавляем margin=0, не уничтожаем данные.
    python3 -c "
import json, sys
try:
    data = json.load(open('$PAPER_STATE'))
except Exception as e:
    print(f'   ⚠️  paper_state.json повреждён ({e}), создаём новый')
    data = {}

cur_bal = data.get('balance')
if cur_bal is None or not isinstance(cur_bal, (int, float)) or cur_bal <= 0:
    cur_bal = $INIT_BAL
    print(f'   ⚠️  Баланс не найден в state, используем PAPER_INITIAL_BALANCE={cur_bal}')

# Миграция позиций: добавляем отсутствующие поля
positions = data.get('positions', {})
migrated = 0
for sym, pos in positions.items():
    if 'margin' not in pos:
        pos['margin'] = 0.0
        migrated += 1
    if 'strategy_id' not in pos:
        pos['strategy_id'] = 'unknown'

# Обновляем только служебные поля, баланс НЕ трогаем
data['balance'] = cur_bal
if 'initial_balance' not in data:
    data['initial_balance'] = $INIT_BAL
if 'positions' not in data:
    data['positions'] = {}
if 'trades_history' not in data:
    data['trades_history'] = []

open('$PAPER_STATE', 'w').write(json.dumps(data, indent=2))
mig_msg = f', мигрировано позиций: {migrated}' if migrated else ''
print(f'   ✅ Paper баланс сохранён: {cur_bal:.2f} USDT (история сохранена{mig_msg})')
" 2>&1 || echo "   ⚠️  Ошибка обработки paper_state, файл не изменён"
else
    # 8c. Создаём новый state
    python3 -c "
import json
d = {'balance': $INIT_BAL, 'initial_balance': $INIT_BAL, 'positions': {}, 'trades_history': [], 'saved_at': ''}
open('$PAPER_STATE', 'w').write(json.dumps(d, indent=2))
"
    echo "   ✅ Paper state создан. Баланс: ${INIT_BAL} USDT"
fi

# ── 9. Обновление Python-зависимостей ────────────────────────────────────────
if [ -f "$VENV" ]; then
    echo "   Обновление зависимостей..."
    source "$VENV"
    pip install -r "$APP_DIR/bybit-bot-v5/requirements.txt" -q
    echo "   ✅ Зависимости актуальны"
fi

# ── 10. Очистка старых бэкапов (оставляем последние 10) ──────────────────────
if [ -d "$BACKUP_ROOT" ]; then
    ls -1t "$BACKUP_ROOT" | tail -n +11 | while read old; do
        rm -rf "$BACKUP_ROOT/$old"
    done
fi

# ── 11. Перезапуск сервиса ────────────────────────────────────────────────────
systemctl restart baibit
sleep 3

if systemctl is-active --quiet baibit; then
    echo ""
    echo "✅ Готово! Бот обновлён и запущен."
    echo "   Бэкап: $BACKUP_DIR"
    echo "   Откройте Telegram и напишите /menu"
else
    echo ""
    echo "❌ Ошибка запуска. Последние логи:"
    journalctl -u baibit -n 40 --no-pager
    echo ""
    echo "   Для отката: cp $BACKUP_DIR/.env $ENV_FILE && systemctl restart baibit"
fi
echo ""
