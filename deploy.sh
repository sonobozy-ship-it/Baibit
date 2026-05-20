#!/bin/bash
# ╔══════════════════════════════════════════════════════╗
# ║          BAIBIT — ДЕПЛОЙ НА VPS ОДНОЙ КОМАНДОЙ      ║
# ║  Использование:  ./deploy.sh                         ║
# ║  С паролем:      ./deploy.sh --ask-pass              ║
# ╚══════════════════════════════════════════════════════╝

set -e

# ── Настройки ────────────────────────────────────────────
VPS_HOST="31.172.77.105"
VPS_USER="root"
VPS_PORT="22"
REMOTE_DIR="/home/user/Baibit"
LOCAL_SRC="$(cd "$(dirname "$0")" && pwd)/bybit-bot-v5/backend"
ARCHIVE="/tmp/baibit-update.zip"
LOG_FILE="/tmp/baibit.log"
APP_PORT="8000"

# ── Цвета ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step()  { echo -e "\n${CYAN}▶ $1${NC}"; }
ok()    { echo -e "${GREEN}✅ $1${NC}"; }
warn()  { echo -e "${YELLOW}⚠  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# ── SSH параметры ────────────────────────────────────────
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $VPS_PORT"
if [[ "$1" == "--ask-pass" ]]; then
    SSH_OPTS="$SSH_OPTS -o PasswordAuthentication=yes"
    SCP_OPTS="-P $VPS_PORT -o StrictHostKeyChecking=no"
else
    SCP_OPTS="-P $VPS_PORT -o StrictHostKeyChecking=no"
fi

SSH="ssh $SSH_OPTS $VPS_USER@$VPS_HOST"
SCP="scp $SCP_OPTS"

# ════════════════════════════════════════════════════════
echo -e "\n${CYAN}╔════════════════════════════════════════╗"
echo -e "║   🚀  BAIBIT DEPLOY  →  $VPS_HOST  ║"
echo -e "╚════════════════════════════════════════╝${NC}"

# ── 1. Проверка SSH ──────────────────────────────────────
step "Проверка подключения к VPS..."
$SSH "echo connected" > /dev/null 2>&1 || error "Нет SSH доступа к $VPS_HOST. Настрой ключ: ssh-copy-id $VPS_USER@$VPS_HOST"
ok "SSH соединение установлено"

# ── 2. Создание архива ───────────────────────────────────
step "Создание архива проекта..."
rm -f "$ARCHIVE"
zip -r "$ARCHIVE" "$LOCAL_SRC" \
    --exclude "*/\__pycache__/*" \
    --exclude "*/*.pyc" \
    --exclude "*/data/ml_data.db" \
    --exclude "*/data/boost_session.json" \
    --exclude "*/data/*.pkl" \
    --exclude "*/data/*.parquet" \
    --exclude "*/.env" \
    -q
ARCHIVE_SIZE=$(du -sh "$ARCHIVE" | cut -f1)
ok "Архив создан: $ARCHIVE ($ARCHIVE_SIZE)"

# ── 3. Загрузка на VPS ───────────────────────────────────
step "Загрузка архива на VPS..."
$SCP "$ARCHIVE" "$VPS_USER@$VPS_HOST:/tmp/" || error "Ошибка загрузки файла"
ok "Архив загружен на VPS"

# ── 4. Распаковка и обновление ───────────────────────────
step "Распаковка и обновление файлов на VPS..."
$SSH bash << EOF
set -e
cd /tmp

# Распаковка с заменой файлов
unzip -o baibit-update.zip -d "$REMOTE_DIR/" > /dev/null 2>&1
echo "Распаковка завершена"

# Установка/обновление зависимостей из requirements.txt
REQ="$REMOTE_DIR/bybit-bot-v5/requirements.txt"
if [ -f "\$REQ" ]; then
    pip install -q -r "\$REQ" 2>/dev/null || pip install -q -r "\$REQ" --break-system-packages 2>/dev/null || true
else
    pip install -q anthropic openai httpx fastapi uvicorn python-dotenv openpyxl pandas numpy 2>/dev/null || true
fi
echo "Зависимости OK"

rm -f /tmp/baibit-update.zip
EOF
ok "Файлы обновлены"

# ── 5. Перезапуск бота ───────────────────────────────────
step "Перезапуск бота..."
$SSH bash << EOF
# systemd
if systemctl is-active --quiet baibit 2>/dev/null; then
    systemctl restart baibit
    echo "RESTART:systemd"

# pm2
elif pm2 list 2>/dev/null | grep -q baibit; then
    pm2 restart baibit
    echo "RESTART:pm2"

# uvicorn напрямую
else
    pkill -f "uvicorn main:app" 2>/dev/null || true
    sleep 1
    cd "$REMOTE_DIR/bybit-bot-v5/backend"
    nohup uvicorn main:app --host 0.0.0.0 --port $APP_PORT > "$LOG_FILE" 2>&1 &
    sleep 2
    if pgrep -f "uvicorn main:app" > /dev/null; then
        echo "RESTART:uvicorn PID \$(pgrep -f 'uvicorn main:app')"
    else
        echo "RESTART:failed"
        cat "$LOG_FILE" | tail -20
    fi
fi
EOF

# ── 6. Проверка статуса ──────────────────────────────────
step "Проверка статуса..."
sleep 2
STATUS=$($SSH "curl -s --max-time 5 http://localhost:$APP_PORT/health 2>/dev/null || echo 'no_response'")

if echo "$STATUS" | grep -q "no_response"; then
    warn "Бот запущен, но /health не отвечает (возможно другой endpoint)"
    $SSH "tail -5 $LOG_FILE 2>/dev/null || true"
else
    ok "Бот отвечает: $STATUS"
fi

# ── Итог ─────────────────────────────────────────────────
echo -e "\n${GREEN}╔════════════════════════════════════════╗"
echo -e "║   ✅  ДЕПЛОЙ ЗАВЕРШЁН УСПЕШНО          ║"
echo -e "╚════════════════════════════════════════╝${NC}"
echo -e "  VPS:    $VPS_HOST:$APP_PORT"
echo -e "  Логи:   ssh $VPS_USER@$VPS_HOST 'tail -f $LOG_FILE'"
echo -e "  Статус: ssh $VPS_USER@$VPS_HOST 'curl -s localhost:$APP_PORT/health'"
echo ""
