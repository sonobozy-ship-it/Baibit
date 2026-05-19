#!/bin/bash
# Обновление бота без затирания ключей
set -e

APP_DIR="/opt/baibit"
BRANCH="claude/analyze-repository-files-TvlQi"

echo ""
echo "🔄 Обновление Baibit..."

cd "$APP_DIR"
git fetch origin "$BRANCH" -q
git reset --hard "origin/$BRANCH" -q

systemctl restart baibit
sleep 3

if systemctl is-active --quiet baibit; then
    echo "✅ Всё готово! Бот обновлён и запущен."
    echo "   Откройте Telegram и напишите /menu"
else
    echo "❌ Ошибка запуска. Логи:"
    journalctl -u baibit -n 20 --no-pager
fi
echo ""
