package com.example;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class Database {
    // URL для подключения к базе данных SQLite
    private static final String URL = "jdbc:sqlite:dailyplanner.db";
    
    // Статический блок инициализации для загрузки драйвера SQLite
    static {
        try {
            // Попытка загрузить драйвер JDBC для SQLite
            Class.forName("org.sqlite.JDBC");
        } catch (ClassNotFoundException e) {
            // Если драйвер не найден, выводим ошибку
            e.printStackTrace();
        }
    }
    
    // Метод для подключения к базе данных
    public static Connection connect() throws SQLException {
        // Подключаемся к базе данных SQLite с помощью URL
        return DriverManager.getConnection(URL);
    }
    
    // Метод для инициализации базы данных и создания необходимых таблиц
    public static void initialize() {
        // Попытка установить подключение и выполнить SQL-запросы
        try (Connection conn = connect(); // Открытие подключения
             Statement stmt = conn.createStatement()) { // Создание объекта Statement для выполнения запросов
            
            // SQL-запрос для создания таблицы задач, если она не существует
            String createTasksTable = "CREATE TABLE IF NOT EXISTS tasks (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT," + // Уникальный идентификатор задачи
                    "title TEXT NOT NULL," + // Заголовок задачи (обязательное поле)
                    "description TEXT," + // Описание задачи (необязательное поле)
                    "due_date TEXT," + // Дата выполнения задачи (необязательное поле)
                    "priority INTEGER," + // Приоритет задачи (необязательное поле)
                    "completed BOOLEAN DEFAULT 0)"; // Статус завершенности (по умолчанию 0 — не завершено)
            
            // SQL-запрос для создания таблицы заметок, если она не существует
            String createNotesTable = "CREATE TABLE IF NOT EXISTS notes (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT," + // Уникальный идентификатор заметки
                    "title TEXT NOT NULL," + // Заголовок заметки (обязательное поле)
                    "content TEXT," + // Содержание заметки (необязательное поле)
                    "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"; // Дата и время создания заметки (по умолчанию текущее время)

            // Выполняем запросы для создания таблиц в базе данных
            stmt.execute(createTasksTable); // Создаем таблицу задач
            stmt.execute(createNotesTable); // Создаем таблицу заметок
            
        } catch (SQLException e) {
            // Если возникает ошибка при подключении или выполнении запроса, выводим сообщение об ошибке
            System.err.println("Ошибка при инициализации базы данных: " + e.getMessage());
        }
    }
}
