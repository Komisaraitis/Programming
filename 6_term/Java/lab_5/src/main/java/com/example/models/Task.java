package com.example.models;

import javafx.beans.property.*;
import java.time.LocalDate;

public class Task {
    private final IntegerProperty id = new SimpleIntegerProperty();
    private final StringProperty title = new SimpleStringProperty();
    private final StringProperty description = new SimpleStringProperty();
    private final ObjectProperty<LocalDate> dueDate = new SimpleObjectProperty<>();
    private final IntegerProperty priority = new SimpleIntegerProperty();


    // Конструкторы
    public Task() {}

    public Task(String title, String description, LocalDate dueDate, int priority) {
        setTitle(title);
        setDescription(description);
        setDueDate(dueDate);
        setPriority(priority);

    }

    // Геттеры свойств (для JavaFX)
    public StringProperty titleProperty() { return title; }
    public StringProperty descriptionProperty() { return description; }
    public ObjectProperty<LocalDate> dueDateProperty() { return dueDate; }
    public IntegerProperty priorityProperty() { return priority; }


    // Метод для отображения даты в таблице
    public StringProperty dueDateStringProperty() {
        return new SimpleStringProperty(getDueDate().toString());
    }

    // Стандартные геттеры и сеттеры
    public int getId() { return id.get(); }
    public void setId(int id) { this.id.set(id); }
    
    public String getTitle() { return title.get(); }
    public void setTitle(String title) { this.title.set(title); }
    
    public String getDescription() { return description.get(); }
    public void setDescription(String description) { this.description.set(description); }
    
    public LocalDate getDueDate() { return dueDate.get(); }
    public void setDueDate(LocalDate dueDate) { this.dueDate.set(dueDate); }
    
    public int getPriority() { return priority.get(); }
    public void setPriority(int priority) { this.priority.set(priority); }
    

}