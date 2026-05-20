#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║     BAIBIT — обновление бота прямо на VPS через git pull    ║
# ║  Запускать на VPS: bash update_vps.sh                       ║
# ╚══════════════════════════════════════════════════════════════╝
set -e

REPO_DIR="/home/user/Baibit"
BACKEND_DIR="$REPO_DIR/bybit-bot-v5/backend"
BRANCH="${1:-main}"     # ветка передаётся первым аргументом, по умолчанию main

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${CYAN}▶ $1${NC}"; }
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }

step "Обновление кода ($BRANCH)..."
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"
ok "Код обновлён до: $(git log -1 --pretty='%h %s')"

step "Обновление зависимостей Python..."
REQ="$REPO_DIR/bybit-bot-v5/requirements.txt"
if command -v pip3 &>/dev/null; then
    pip3 install -q -r "$REQ" 2>/dev/null \
        || pip3 install -q -r "$REQ" --break-system-packages 2>/dev/null \
        || true
else
    pip install -q -r "$REQ" 2>/dev/null \
        || pip install -q -r "$REQ" --break-system-packages 2>/dev/null \
        || true
fi
ok "Зависимости обновлены"

step "Перезапуск бота..."
if systemctl is-active --quiet baibit 2>/dev/null; then
    systemctl restart baibit
    sleep 2
    systemctl is-active --quiet baibit && ok "Сервис baibit перезапущен (systemd)" || err "Сервис не поднялся"
elif pm2 list 2>/dev/null | grep -q baibit; then
    pm2 restart baibit
    ok "Перезапущен через pm2"
else
    # Прямой запуск uvicorn
    pkill -f "uvicorn main:app" 2>/dev/null || true
    sleep 1
    cd "$BACKEND_DIR"
    LOG="/tmp/baibit.log"
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 > "$LOG" 2>&1 &
    sleep 3
    if pgrep -f "uvicorn main:app" > /dev/null; then
        ok "uvicorn запущен (PID: $(pgrep -f 'uvicorn main:app'))"
    else
        echo "=== Последние ошибки ==="
        tail -30 "$LOG"
        err "uvicorn не запустился"
    fi
fi

step "Проверка /health..."
sleep 2
HEALTH=$(curl -s --max-time 5 http://localhost:8000/health 2>/dev/null || echo "no_response")
if echo "$HEALTH" | grep -q "no_response"; then
    echo -e "${CYAN}⚠  /health не отвечает — возможно другой порт или endpoint${NC}"
    echo "Логи: journalctl -u baibit -n 30 --no-pager"
else
    ok "Бот отвечает: $HEALTH"
fi

echo -e "\n${GREEN}╔════════════════════════════════════╗"
echo -e "║   ✅  ОБНОВЛЕНИЕ ЗАВЕРШЕНО          ║"
echo -e "╚════════════════════════════════════╝${NC}"
echo -e "  Коммит: $(git log -1 --pretty='%h %s')"
echo -e "  Логи:   journalctl -u baibit -f"
echo ""
