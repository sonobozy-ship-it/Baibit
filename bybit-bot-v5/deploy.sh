#!/bin/bash
# Full VPS deployment script for Baibit trading bot
# Usage: bash deploy.sh
# Tested on Ubuntu 22.04 / Debian 12

set -euo pipefail

APP_DIR="/opt/baibit"
SERVICE_NAME="baibit"
PYTHON="python3"

echo "=== Baibit VPS Deployment ==="

# ── System dependencies ────────────────────────────────────────
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    mysql-server \
    git curl build-essential

# ── MySQL setup ────────────────────────────────────────────────
echo "[2/7] Configuring MySQL..."
systemctl enable --now mysql

read -rp "Enter MySQL root password (leave blank if none): " MYSQL_ROOT_PW
MYSQL_CMD="mysql -u root"
if [ -n "$MYSQL_ROOT_PW" ]; then
    MYSQL_CMD="mysql -u root -p$MYSQL_ROOT_PW"
fi

read -rp "Enter password for baibit DB user: " BAIBIT_PW
$MYSQL_CMD -e "
CREATE DATABASE IF NOT EXISTS baibit CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'baibit'@'localhost' IDENTIFIED BY '$BAIBIT_PW';
GRANT ALL PRIVILEGES ON baibit.* TO 'baibit'@'localhost';
FLUSH PRIVILEGES;
"
echo "MySQL: database 'baibit' and user 'baibit'@'localhost' created."

# ── App directory ──────────────────────────────────────────────
echo "[3/7] Setting up app directory..."
mkdir -p "$APP_DIR"
# Copy files if running from source directory, else clone
if [ -f "requirements.txt" ]; then
    cp -r . "$APP_DIR/"
else
    read -rp "Enter git repo URL: " REPO_URL
    git clone "$REPO_URL" "$APP_DIR"
fi

# ── Python environment ─────────────────────────────────────────
echo "[4/7] Creating Python virtual environment..."
cd "$APP_DIR"
$PYTHON -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "Python deps installed."

# ── Environment file ───────────────────────────────────────────
echo "[5/7] Creating .env file..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    # Inject MySQL password
    sed -i "s/strong_password_here/$BAIBIT_PW/" "$APP_DIR/.env"
    echo ""
    echo ">>> EDIT $APP_DIR/.env and fill in your API keys before starting the bot <<<"
    echo ""
fi

# ── Data directories ───────────────────────────────────────────
mkdir -p "$APP_DIR/backend/data/candles" \
         "$APP_DIR/backend/data/features" \
         "$APP_DIR/backend/data/models" \
         "$APP_DIR/backend/logs"

# ── Systemd service ────────────────────────────────────────────
echo "[6/7] Creating systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=Baibit Trading Bot
After=network.target mysql.service
Requires=mysql.service

[Service]
Type=simple
User=root
WorkingDirectory=${APP_DIR}/backend
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

echo "[7/7] Deployment complete!"
echo ""
echo "Next steps:"
echo "  1. Edit ${APP_DIR}/.env with your Bybit/Anthropic/Telegram API keys"
echo "  2. Start the bot:  systemctl start $SERVICE_NAME"
echo "  3. View logs:      journalctl -u $SERVICE_NAME -f"
echo "  4. API access:     http://YOUR_VPS_IP:8000/docs"
echo ""
