#!/bin/bash
# Baibit — быстрый деплой на VPS (paper-режим)
set -e

APP_DIR="/opt/baibit"
REPO="https://github.com/sonobozy-ship-it/baibit.git"
BRANCH="claude/analyze-repository-files-TvlQi"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     Baibit Trading Bot — Install     ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Системные пакеты ────────────────────────────────────────
echo "[1/6] Установка системных пакетов..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl build-essential mysql-server

# ── 2. MySQL ───────────────────────────────────────────────────
echo "[2/6] Настройка MySQL..."
systemctl enable --now mysql 2>/dev/null || true
mysql -u root -e "
CREATE DATABASE IF NOT EXISTS baibit CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'baibit'@'localhost' IDENTIFIED BY '73501505aAaA!';
GRANT ALL PRIVILEGES ON baibit.* TO 'baibit'@'localhost';
FLUSH PRIVILEGES;
" 2>/dev/null || true
echo "    MySQL готов"

# ── 3. Код ────────────────────────────────────────────────────
echo "[3/6] Загрузка кода..."
rm -rf "$APP_DIR"
git clone --branch "$BRANCH" "$REPO" "$APP_DIR" -q
cd "$APP_DIR/bybit-bot-v5"

# ── 4. Python окружение ────────────────────────────────────────
echo "[4/6] Установка Python-зависимостей..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "    Python-зависимости установлены"

# ── 5. .env файл ──────────────────────────────────────────────
echo "[5/6] Настройка конфигурации..."
mkdir -p backend/data/candles backend/data/features backend/data/models backend/logs

# Не перезаписываем .env если уже существует (сохраняем ключи)
if [ -f backend/.env ]; then
    echo "    .env уже существует — ключи сохранены"
else
cat > backend/.env << 'ENVEOF'
BYBIT_API_KEY=
BYBIT_API_SECRET=
BYBIT_TESTNET=false

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OLLAMA_BASE_URL=

TELEGRAM_TOKEN=8865970945:AAEeRQMJ2YsHoRULyM9a727nygg3TphjSqc
TELEGRAM_CHAT_ID=872119694

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=baibit
MYSQL_PASSWORD=73501505aAaA!
MYSQL_DATABASE=baibit

CRYPTOPANIC_API_KEY=

SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
PAPER_TRADING=true
AUTO_START=true
MAX_LEVERAGE=5
ADAPTIVE_WINDOW=30
ENVEOF
fi

echo "    Конфигурация создана"

# ── 6. Systemd сервис ─────────────────────────────────────────
echo "[6/6] Создание системного сервиса..."
cat > /etc/systemd/system/baibit.service << SVCEOF
[Unit]
Description=Baibit Trading Bot
After=network.target mysql.service
Requires=mysql.service

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}/bybit-bot-v5/backend
EnvironmentFile=${APP_DIR}/bybit-bot-v5/backend/.env
ExecStart=${APP_DIR}/bybit-bot-v5/venv/bin/python runner.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=baibit

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable baibit
systemctl restart baibit

sleep 3
echo ""
if systemctl is-active --quiet baibit; then
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║          ✅  Всё готово! Baibit запущен.             ║"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  Telegram:    откройте бота и напишите /menu         ║"
    echo "║  Логи:        journalctl -u baibit -f                ║"
    echo "║  Обновление:  bash /opt/baibit/bybit-bot-v5/update.sh║"
    echo "╚══════════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  ❌  Сервис не запустился. Проверьте логи:           ║"
    echo "║  journalctl -u baibit -n 30 --no-pager              ║"
    echo "╚══════════════════════════════════════════════════════╝"
fi
echo ""
