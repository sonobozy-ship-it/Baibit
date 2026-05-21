#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  sync_git_mac.sh — GitHub ↔ Mac  /Documents/New project 3       ║
# ║                                                                  ║
# ║  ./sync_git_mac.sh pull          — скачать репо в папку на Mac  ║
# ║  ./sync_git_mac.sh push          — запушить изменения на GitHub  ║
# ║  ./sync_git_mac.sh push "msg"    — запушить с комментарием       ║
# ║  ./sync_git_mac.sh status        — показать статус репо          ║
# ╚══════════════════════════════════════════════════════════════════╝
set -e

# ════════════════════════════════════════════════════════════════════
# ⚙️  НАСТРОЙКИ — измените если нужно
# ════════════════════════════════════════════════════════════════════
REPO_URL="https://github.com/sonobozy-ship-it/baibit.git"
BRANCH="main"
LOCAL_DEST="$HOME/Documents/New project 3"
# ════════════════════════════════════════════════════════════════════

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

step() { echo -e "\n${CYAN}▶ $1${NC}"; }
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }

is_git_repo() {
    [ -d "$LOCAL_DEST/.git" ]
}

# ── PULL: GitHub → Mac ──────────────────────────────────────────────
cmd_pull() {
    echo -e "\n${BOLD}📥 PULL: GitHub → Mac${NC}"
    echo -e "   Репо:  ${CYAN}$REPO_URL${NC}"
    echo -e "   Ветка: ${CYAN}$BRANCH${NC}"
    echo -e "   Куда:  ${CYAN}$LOCAL_DEST${NC}"

    if is_git_repo; then
        # Репо уже есть — обновляем
        step "Обновление репозитория (git pull)..."
        git -C "$LOCAL_DEST" fetch origin
        git -C "$LOCAL_DEST" checkout "$BRANCH" 2>/dev/null || \
            git -C "$LOCAL_DEST" checkout -b "$BRANCH" "origin/$BRANCH"
        git -C "$LOCAL_DEST" pull origin "$BRANCH"
        ok "Репо обновлён"
    else
        # Первый раз — клонируем
        if [ -d "$LOCAL_DEST" ] && [ "$(ls -A "$LOCAL_DEST" 2>/dev/null)" ]; then
            warn "Папка $LOCAL_DEST существует и не пустая"
            echo -e "   Клонируем поверх (git clone --branch)..."
            mkdir -p "$LOCAL_DEST"
        else
            mkdir -p "$LOCAL_DEST"
        fi

        step "Клонирование репозитория..."
        git clone --branch "$BRANCH" "$REPO_URL" "$LOCAL_DEST"
        ok "Репо клонирован в $LOCAL_DEST"
    fi

    # Показываем последние коммиты
    echo -e "\n${BOLD}Последние коммиты:${NC}"
    git -C "$LOCAL_DEST" log --oneline -5

    echo -e "\n${GREEN}${BOLD}╔════════════════════════════════════════╗"
    echo -e "║   ✅  PULL ЗАВЕРШЁН                     ║"
    echo -e "╚════════════════════════════════════════╝${NC}"
    echo -e "   Папка: ${CYAN}$LOCAL_DEST${NC}\n"
}

# ── PUSH: Mac → GitHub ──────────────────────────────────────────────
cmd_push() {
    COMMIT_MSG="${2:-update: изменения с Mac}"

    echo -e "\n${BOLD}📤 PUSH: Mac → GitHub${NC}"
    echo -e "   Папка:  ${CYAN}$LOCAL_DEST${NC}"
    echo -e "   Ветка:  ${CYAN}$BRANCH${NC}"

    if ! is_git_repo; then
        err "Папка $LOCAL_DEST не является git-репозиторием. Сначала выполните: ./sync_git_mac.sh pull"
    fi

    cd "$LOCAL_DEST"

    # Проверяем, есть ли что коммитить
    CHANGED=$(git status --porcelain 2>/dev/null)
    if [ -z "$CHANGED" ]; then
        warn "Нет изменений для коммита"
        git log --oneline -3
        exit 0
    fi

    echo -e "\n${BOLD}Изменения:${NC}"
    git status --short

    step "Добавление файлов в коммит..."
    # Добавляем всё, кроме чувствительных файлов
    git add --all
    # Убираем .env если случайно попал
    git reset HEAD -- "**/.env" ".env" 2>/dev/null || true

    step "Создание коммита..."
    echo -e "   Сообщение: ${CYAN}$COMMIT_MSG${NC}"
    git commit -m "$COMMIT_MSG" || { warn "Нечего коммитить"; exit 0; }

    step "Push на GitHub (ветка $BRANCH)..."
    git push -u origin "$BRANCH"
    ok "Изменения отправлены на GitHub"

    echo -e "\n${BOLD}Последние коммиты:${NC}"
    git log --oneline -5

    echo -e "\n${GREEN}${BOLD}╔════════════════════════════════════════╗"
    echo -e "║   ✅  PUSH ЗАВЕРШЁН                     ║"
    echo -e "╚════════════════════════════════════════╝${NC}"
    echo -e "   Репо:  ${CYAN}$REPO_URL${NC}"
    echo -e "   Ветка: ${CYAN}$BRANCH${NC}\n"

    echo -e "${YELLOW}💡 Для деплоя на VPS:${NC}"
    echo -e "   ${CYAN}./sync_to_vps.sh${NC}"
    echo -e "   ИЛИ на самом VPS:"
    echo -e "   ${CYAN}ssh root@31.172.77.105 'cd /opt/baibit && git pull && systemctl restart baibit'${NC}\n"
}

# ── STATUS ──────────────────────────────────────────────────────────
cmd_status() {
    echo -e "\n${BOLD}📊 Статус репозитория${NC}"
    echo -e "   Папка: ${CYAN}$LOCAL_DEST${NC}\n"

    if ! is_git_repo; then
        warn "Репозиторий не клонирован. Выполните: ./sync_git_mac.sh pull"
        exit 0
    fi

    cd "$LOCAL_DEST"
    echo -e "${BOLD}Ветка:${NC} $(git branch --show-current)"
    echo -e "${BOLD}Последний коммит:${NC} $(git log --oneline -1)"
    echo ""

    CHANGED=$(git status --porcelain 2>/dev/null)
    if [ -z "$CHANGED" ]; then
        ok "Нет несохранённых изменений"
    else
        echo -e "${BOLD}Изменённые файлы:${NC}"
        git status --short
    fi

    echo -e "\n${BOLD}Последние 5 коммитов:${NC}"
    git log --oneline -5
    echo ""
}

# ── Точка входа ──────────────────────────────────────────────────────
case "$1" in
    pull)   cmd_pull "$@" ;;
    push)   cmd_push "$@" ;;
    status) cmd_status ;;
    *)
        echo -e "\n${BOLD}sync_git_mac.sh — GitHub ↔ Mac /Documents/New project 3${NC}\n"
        echo -e "  ${CYAN}./sync_git_mac.sh pull${NC}           — скачать репо на Mac"
        echo -e "  ${CYAN}./sync_git_mac.sh push${NC}           — запушить изменения на GitHub"
        echo -e "  ${CYAN}./sync_git_mac.sh push \"сообщение\"${NC} — запушить с комментарием"
        echo -e "  ${CYAN}./sync_git_mac.sh status${NC}         — статус репо"
        echo ""
        echo -e "  ${BOLD}Типичный рабочий процесс:${NC}"
        echo -e "    1. ${CYAN}./sync_git_mac.sh pull${NC}       — обновить с GitHub"
        echo -e "    2. ... внести изменения в ~/Documents/New project 3 ..."
        echo -e "    3. ${CYAN}./sync_git_mac.sh push \"fix: что изменил\"${NC}"
        echo -e "    4. ${CYAN}./sync_to_vps.sh${NC}             — задеплоить на VPS"
        echo ""
        ;;
esac
