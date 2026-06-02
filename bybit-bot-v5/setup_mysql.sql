-- Run as MySQL root: mysql -u root -p < setup_mysql.sql

CREATE DATABASE IF NOT EXISTS baibit
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'baibit'@'localhost'
    IDENTIFIED BY 'change_me_strong_password';

GRANT ALL PRIVILEGES ON baibit.* TO 'baibit'@'localhost';
FLUSH PRIVILEGES;

SELECT 'MySQL setup complete. Update MYSQL_PASSWORD in your .env file.' AS status;
