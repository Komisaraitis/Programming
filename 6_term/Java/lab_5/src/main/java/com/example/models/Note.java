package com.example.models;

import javafx.beans.property.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Note {
    // Свойства для хранения данных заметки
    private final IntegerProperty id = new SimpleIntegerProperty(); // Идентификатор заметки
    private final StringProperty title = new SimpleStringProperty(); // Заголовок заметки
    private final StringProperty content = new SimpleStringProperty(); // Содержание заметки
    private final ObjectProperty<LocalDateTime> createdAt = new SimpleObjectProperty<>(); // Дата и время создания заметки

    // Форматтер для отображения даты и времени в виде строки
    private static final DateTimeFormatter DISPLAY_FORMATTER = 
        DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm");

    // Конструкторы
    public Note() {
        // Пустой конструктор
    }

    public Note(String title, String content) {
        setTitle(title); // Устанавливаем заголовок
        setContent(content); // Устанавливаем содержание
        setCreatedAt(LocalDateTime.now()); // Устанавливаем текущую дату и время
    }

    // Property геттеры — используются для получения привязанных свойств JavaFX
    public IntegerProperty idProperty() {
        return id; // Возвращаем свойство id
    }

    public StringProperty titleProperty() {
        return title; // Возвращаем свойство title
    }

    public StringProperty contentProperty() {
        return content; // Возвращаем свойство content
    }

    public ObjectProperty<LocalDateTime> createdAtProperty() {
        return createdAt; // Возвращаем свойство createdAt
    }

    // Стандартные геттеры — возвращают значения свойств
    public int getId() {
        return id.get(); // Возвращаем значение id
    }

    public String getTitle() {
        return title.get(); // Возвращаем значение title
    }

    public String getContent() {
        return content.get(); // Возвращаем значение content
    }

    public LocalDateTime getCreatedAt() {
        return createdAt.get(); // Возвращаем значение createdAt
    }

    // Стандартные сеттеры — устанавливают значения свойств
    public void setId(int id) {
        this.id.set(id); // Устанавливаем значение id
    }

    public void setTitle(String title) {
        this.title.set(title); // Устанавливаем значение title
    }

    public void setContent(String content) {
        this.content.set(content); // Устанавливаем значение content
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt.set(createdAt); // Устанавливаем значение createdAt
    }

    // Метод для преобразования заметки в строку, используемую в ListView
    @Override
    public String toString() {
        // Форматируем строку, показывая заголовок, содержание и дату
        return String.format("%s\n%s\n(%s)", 
            getTitle(), // Заголовок
            getContent(), // Содержание
            getCreatedAt().format(DISPLAY_FORMATTER)); // Отформатированная дата и время создания
    }

    // Метод для получения отформатированной даты и времени в виде строки
    public String getFormattedCreatedAt() {
        return getCreatedAt().format(DISPLAY_FORMATTER); // Возвращаем отформатированную строку даты
    }

    // Метод для сравнения двух заметок по их id
    @Override
    public boolean equals(Object o) {
        if (this == o) return true; // Если ссылки одинаковые, то объекты одинаковые
        if (o == null || getClass() != o.getClass()) return false; // Проверяем, что объект не null и того же типа
        Note note = (Note) o; // Приводим объект к типу Note
        return getId() == note.getId(); // Сравниваем id
    }

    // Метод для вычисления хэш-кода заметки на основе ее id
    @Override
    public int hashCode() {
        return Integer.hashCode(getId()); // Хэш-код на основе id
    }
}
