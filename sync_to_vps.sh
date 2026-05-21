#!/bin/bash
# PC → VPS: отправить изменения на сервер и перезапустить бота
set -e

VPS="root@31.172.77.105"
VPS_PATH="/opt/baibit"
LOCAL="${1:-$HOME/Documents/New project 3}"

echo "📤 Отправка изменений на VPS..."
echo "   Откуда: $LOCAL"
echo "   Куда:   $VPS:$VPS_PATH"
echo ""

rsync -avz --progress \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='venv/' \
  --exclude='*.log' \
  --exclude='.env' \
  --exclude='*.db' \
  --exclude='*.sql' \
  --exclude='data/' \
  --exclude='logs/' \
  "$LOCAL/" "$VPS:$VPS_PATH/"

echo ""
echo "🔄 Перезапуск бота..."
ssh "$VPS" "systemctl restart baibit && sleep 2 && systemctl is-active baibit"

echo ""
echo "✅ Готово! Изменения на VPS применены."
