#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  db_sync.sh — полная синхронизация Baibit: VPS ↔ PC             ║
# ║                                                                  ║
# ║  ./db_sync.sh pull     — скачать всё с VPS на PC                ║
# ║  ./db_sync.sh push     — залить всё с PC на VPS                 ║
# ║  ./db_sync.sh pull db  — только база данных                     ║
# ║  ./db_sync.sh pull data— только data/ (paper, модели)           ║
# ║  ./db_sync.sh list     — список локальных бэкапов               ║
# ╚══════════════════════════════════════════════════════════════════╝
set -e

# ── Настройки ────────────────────────────────────────────────────────
VPS_HOST="root@31.172.77.105"
VPS_APP="/opt/baibit"
VPS_BACKEND="$VPS_APP/bybit-bot-v5/backend"
VPS_ENV="$VPS_BACKEND/.env"

# Локальный репозиторий (папка где лежит этот скрипт)
LOCAL_REPO="$(cd "$(dirname "$0")" && pwd)"
LOCAL_BACKUP_ROOT="$HOME/baibit_backups"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

step()  { echo -e "\n${CYAN}▶ $1${NC}"; }
ok()    { echo -e "${GREEN}✅ $1${NC}"; }
warn()  { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()   { echo -e "${RED}❌ $1${NC}"; exit 1; }
info()  { echo -e "   $1"; }

# ── Получение MySQL-реквизитов с VPS ─────────────────────────────────
get_mysql_creds_vps() {
    VPS_CREDS=$(ssh "$VPS_HOST" "
        set -a; source $VPS_ENV 2>/dev/null || true; set +a
        echo \${MYSQL_USER:-baibit}
        echo \${MYSQL_PASSWORD:-}
        echo \${MYSQL_DATABASE:-baibit}
        echo \${MYSQL_HOST:-127.0.0.1}
    ")
    MYSQL_USER=$(echo "$VPS_CREDS"     | sed -n '1p')
    MYSQL_PASSWORD=$(echo "$VPS_CREDS" | sed -n '2p')
    MYSQL_DATABASE=$(echo "$VPS_CREDS" | sed -n '3p')
    MYSQL_HOST_DB=$(echo "$VPS_CREDS"  | sed -n '4p')
}

# ── Получение MySQL-реквизитов из локального .env ────────────────────
get_mysql_creds_local() {
    LOCAL_ENV="$LOCAL_REPO/bybit-bot-v5/backend/.env"
    if [ ! -f "$LOCAL_ENV" ] && [ -f "$1/.env" ]; then
        LOCAL_ENV="$1/.env"
    fi
    if [ -f "$LOCAL_ENV" ]; then
        set -a; source "$LOCAL_ENV"; set +a
    fi
    MYSQL_USER="${MYSQL_USER:-baibit}"
    MYSQL_DATABASE="${MYSQL_DATABASE:-baibit}"
    MYSQL_HOST_DB="${MYSQL_HOST:-127.0.0.1}"
}

# ────────────────────────────────────────────────────────────────────
# PULL: VPS → PC
# ────────────────────────────────────────────────────────────────────
cmd_pull() {
    TARGET="$2"   # db | data | code | (пусто = всё)
    BACKUP_DIR="$LOCAL_BACKUP_ROOT/$(date +%Y-%m-%d_%H-%M-%S)"
    mkdir -p "$BACKUP_DIR"

    echo -e "\n${BOLD}📥 PULL: VPS → PC${NC}"
    echo -e "   Хост:  ${CYAN}$VPS_HOST${NC}"
    echo -e "   Куда:  ${CYAN}$BACKUP_DIR${NC}"
    [ -n "$TARGET" ] && echo -e "   Цель:  ${CYAN}$TARGET${NC}"

    # ── 1. База данных ────────────────────────────────────────────────
    if [ -z "$TARGET" ] || [ "$TARGET" = "db" ]; then
        step "Дамп MySQL базы данных..."
        get_mysql_creds_vps
        info "База: $MYSQL_DATABASE @ $MYSQL_HOST_DB (user: $MYSQL_USER)"
        DUMP_FILE="$BACKUP_DIR/db_${MYSQL_DATABASE}.sql"
        ssh "$VPS_HOST" "
            mysqldump -u${MYSQL_USER} -p${MYSQL_PASSWORD} -h${MYSQL_HOST_DB} \
                --single-transaction --routines --triggers \
                ${MYSQL_DATABASE} 2>/dev/null
        " > "$DUMP_FILE"
        SIZE=$(du -sh "$DUMP_FILE" | cut -f1)
        ok "db_${MYSQL_DATABASE}.sql (${SIZE})"
    fi

    # ── 2. Файлы data/ (paper_state, модели, фичи) ───────────────────
    if [ -z "$TARGET" ] || [ "$TARGET" = "data" ]; then
        step "Скачивание data/..."
        mkdir -p "$BACKUP_DIR/data"
        rsync -az --info=progress2 \
            "$VPS_HOST:$VPS_BACKEND/data/" \
            "$BACKUP_DIR/data/" 2>/dev/null || \
        scp -r "$VPS_HOST:$VPS_BACKEND/data/" "$BACKUP_DIR/data/" 2>/dev/null || \
            warn "data/ пуста или недоступна"
        ok "data/ скачана"
    fi

    # ── 3. .env ───────────────────────────────────────────────────────
    if [ -z "$TARGET" ] || [ "$TARGET" = "db" ] || [ "$TARGET" = "data" ]; then
        step "Скачивание .env..."
        scp -q "$VPS_HOST:$VPS_ENV" "$BACKUP_DIR/.env" 2>/dev/null || warn ".env не найден"
        ok ".env сохранён"
    fi

    # ── 4. Код репозитория (VPS → локальный репо) ─────────────────────
    if [ -z "$TARGET" ] || [ "$TARGET" = "code" ]; then
        step "Синхронизация кода VPS → локальный репозиторий..."
        info "Источник: $VPS_APP/ → $LOCAL_REPO/"
        rsync -az --info=progress2 \
            --exclude='.git/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='venv/' \
            --exclude='node_modules/' \
            --exclude='*.log' \
            --exclude='data/' \
            --exclude='.env' \
            "$VPS_HOST:$VPS_APP/" \
            "$LOCAL_REPO/" 2>/dev/null || \
            warn "rsync недоступен, используем scp"
        ok "Код синхронизирован в $LOCAL_REPO/"
    fi

    # ── Мета-файл ────────────────────────────────────────────────────
    cat > "$BACKUP_DIR/backup_info.txt" <<EOF
Date:   $(date)
VPS:    $VPS_HOST
Target: ${TARGET:-all}
EOF

    # Создаём симлинк latest
    ln -sfn "$BACKUP_DIR" "$LOCAL_BACKUP_ROOT/latest"

    echo -e "\n${GREEN}${BOLD}╔════════════════════════════════════════╗"
    echo -e "║   ✅  PULL ЗАВЕРШЁН                     ║"
    echo -e "╚════════════════════════════════════════╝${NC}"
    echo -e "   Бэкап: ${CYAN}$BACKUP_DIR${NC}"
    echo -e "   Файлы: $(ls "$BACKUP_DIR" | tr '\n' ' ')"
    echo ""
}

# ────────────────────────────────────────────────────────────────────
# PUSH: PC → VPS (с заменой файлов)
# ────────────────────────────────────────────────────────────────────
cmd_push() {
    TARGET="$2"   # db | data | code | (пусто = всё)

    # Выбор бэкапа (latest или указанный)
    if [ -L "$LOCAL_BACKUP_ROOT/latest" ]; then
        BACKUP_DIR=$(readlink -f "$LOCAL_BACKUP_ROOT/latest")
    else
        BACKUP_DIR=$(ls -1dt "$LOCAL_BACKUP_ROOT"/*/ 2>/dev/null | head -1)
        BACKUP_DIR="${BACKUP_DIR%/}"
    fi

    if [ -z "$BACKUP_DIR" ] || [ ! -d "$BACKUP_DIR" ]; then
        err "Нет бэкапа. Сначала выполните: ./db_sync.sh pull"
    fi

    echo -e "\n${BOLD}📤 PUSH: PC → VPS (с заменой файлов)${NC}"
    echo -e "   Хост:   ${CYAN}$VPS_HOST${NC}"
    echo -e "   Бэкап:  ${CYAN}$BACKUP_DIR${NC}"
    [ -n "$TARGET" ] && echo -e "   Цель:   ${CYAN}$TARGET${NC}"
    echo ""

    # Подтверждение
    warn "Это перезапишет файлы на VPS!"
    read -r -p "   Продолжить? (y/N) " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Отменено."; exit 0; }

    # Останавливаем бота
    step "Остановка бота на VPS..."
    ssh "$VPS_HOST" "systemctl stop baibit 2>/dev/null || true; sleep 1"
    ok "Бот остановлен"

    # ── 1. База данных ────────────────────────────────────────────────
    if [ -z "$TARGET" ] || [ "$TARGET" = "db" ]; then
        get_mysql_creds_local "$BACKUP_DIR"
        DUMP_FILE=$(ls "$BACKUP_DIR"/db_*.sql 2>/dev/null | head -1)
        if [ -f "$DUMP_FILE" ]; then
            step "Заливка MySQL дампа на VPS..."
            SIZE=$(du -sh "$DUMP_FILE" | cut -f1)
            info "Файл: $(basename "$DUMP_FILE") ($SIZE)"
            scp -q "$DUMP_FILE" "$VPS_HOST:/tmp/baibit_restore.sql"
            ssh "$VPS_HOST" "
                set -a; source $VPS_ENV 2>/dev/null || true; set +a
                mysql -u\${MYSQL_USER:-baibit} -p\${MYSQL_PASSWORD} \
                    -h\${MYSQL_HOST:-127.0.0.1} \${MYSQL_DATABASE:-baibit} \
                    < /tmp/baibit_restore.sql 2>/dev/null
                rm -f /tmp/baibit_restore.sql
            "
            ok "MySQL база восстановлена"
        else
            warn "SQL дамп не найден, пропускаем"
        fi
    fi

    # ── 2. data/ ──────────────────────────────────────────────────────
    if [ -z "$TARGET" ] || [ "$TARGET" = "data" ]; then
        if [ -d "$BACKUP_DIR/data" ]; then
            step "Заливка data/ на VPS (с заменой)..."
            ssh "$VPS_HOST" "mkdir -p $VPS_BACKEND/data"
            rsync -az --delete --info=progress2 \
                "$BACKUP_DIR/data/" \
                "$VPS_HOST:$VPS_BACKEND/data/" 2>/dev/null || \
            scp -r "$BACKUP_DIR/data/." "$VPS_HOST:$VPS_BACKEND/data/"
            ok "data/ загружена"
        else
            warn "data/ не найдена в бэкапе, пропускаем"
        fi
    fi

    # ── 3. Код (локальный репо → VPS, с заменой) ─────────────────────
    if [ -z "$TARGET" ] || [ "$TARGET" = "code" ]; then
        step "Синхронизация кода локальный репо → VPS (с заменой)..."
        info "Источник: $LOCAL_REPO/ → $VPS_APP/"
        rsync -az --delete --info=progress2 \
            --exclude='.git/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='venv/' \
            --exclude='node_modules/' \
            --exclude='*.log' \
            --exclude='data/' \
            --exclude='.env' \
            "$LOCAL_REPO/" \
            "$VPS_HOST:$VPS_APP/" 2>/dev/null || \
            warn "rsync недоступен, файлы не синхронизированы"
        ok "Код синхронизирован на VPS"
    fi

    # Запускаем бота
    step "Запуск бота на VPS..."
    ssh "$VPS_HOST" "systemctl start baibit"
    sleep 3
    STATUS=$(ssh "$VPS_HOST" "systemctl is-active baibit 2>/dev/null || echo inactive")
    if [ "$STATUS" = "active" ]; then
        ok "Бот запущен"
    else
        warn "Бот не запустился. Проверьте: journalctl -u baibit -n 30"
    fi

    echo -e "\n${GREEN}${BOLD}╔════════════════════════════════════════╗"
    echo -e "║   ✅  PUSH ЗАВЕРШЁН                     ║"
    echo -e "╚════════════════════════════════════════╝${NC}"
    echo ""
}

# ── Команда LIST ─────────────────────────────────────────────────────
cmd_list() {
    echo -e "\n${BOLD}📦 Локальные бэкапы:${NC}"
    echo -e "   ${CYAN}$LOCAL_BACKUP_ROOT${NC}\n"
    if [ ! -d "$LOCAL_BACKUP_ROOT" ] || [ -z "$(ls -A "$LOCAL_BACKUP_ROOT" 2>/dev/null)" ]; then
        warn "Бэкапов нет. Выполните: ./db_sync.sh pull"
        return
    fi
    i=1
    for dir in $(ls -1dt "$LOCAL_BACKUP_ROOT"/*/); do
        dir="${dir%/}"
        [ "$dir" = "$LOCAL_BACKUP_ROOT/latest" ] && continue
        name=$(basename "$dir")
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        files=$(ls "$dir" | tr '\n' ' ')
        if [ "$i" = "1" ]; then
            echo -e "  ${GREEN}${BOLD}→ $name${NC}  [${size}]  ← latest"
        else
            echo -e "    $name  [${size}]"
        fi
        echo -e "       ${files}"
        i=$((i + 1))
    done
    echo ""
}

# ── Точка входа ──────────────────────────────────────────────────────
case "$1" in
    pull)   cmd_pull "$@" ;;
    push)   cmd_push "$@" ;;
    list)   cmd_list ;;
    *)
        echo -e "\n${BOLD}db_sync.sh — синхронизация Baibit VPS ↔ PC${NC}\n"
        echo -e "  ${CYAN}./db_sync.sh pull${NC}        — скачать всё с VPS (БД + data + код)"
        echo -e "  ${CYAN}./db_sync.sh pull db${NC}     — только база данных"
        echo -e "  ${CYAN}./db_sync.sh pull data${NC}   — только data/ (paper_state, модели)"
        echo -e "  ${CYAN}./db_sync.sh pull code${NC}   — только код репозитория"
        echo -e ""
        echo -e "  ${CYAN}./db_sync.sh push${NC}        — залить всё на VPS (с заменой файлов)"
        echo -e "  ${CYAN}./db_sync.sh push db${NC}     — только база данных"
        echo -e "  ${CYAN}./db_sync.sh push data${NC}   — только data/"
        echo -e "  ${CYAN}./db_sync.sh push code${NC}   — только код"
        echo -e ""
        echo -e "  ${CYAN}./db_sync.sh list${NC}        — список локальных бэкапов"
        echo ""
        ;;
esac
