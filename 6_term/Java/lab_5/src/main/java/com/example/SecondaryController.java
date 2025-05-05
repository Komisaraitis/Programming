package com.example;

import com.example.dao.NoteDao;
import com.example.models.Note;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;

import java.sql.SQLException;
import java.util.List;

public class SecondaryController {
    // Привязка элементов интерфейса из FXML
    @FXML private ListView<Note> notesListView; // Список для отображения заметок
    @FXML private TextField noteTitleField; // Поле для ввода заголовка заметки
    @FXML private TextArea noteContentArea; // Поле для ввода содержания заметки

    // Объект для работы с заметками в базе данных через NoteDao
    private final NoteDao noteDao = new NoteDao();
    // ObservableList для хранения заметок и их отображения в ListView
    private final ObservableList<Note> notes = FXCollections.observableArrayList();

    // Метод инициализации, который вызывается при старте контроллера
    @FXML
    public void initialize() {
        // Настройка ListView для отображения заметок
        notesListView.setItems(notes); // Привязка списка заметок к ListView
        notesListView.setCellFactory(lv -> new ListCell<Note>() {
            // Настройка отображения заметки в ListView
            @Override
            protected void updateItem(Note note, boolean empty) {
                super.updateItem(note, empty);
                if (empty || note == null) {
                    setText(null); // Если заметка пуста, убираем текст
                    setGraphic(null); // Убираем графику
                } else {
                    // Отображаем заголовок и содержание заметки в виде метки
                    Label label = new Label(note.toString());
                    label.setWrapText(true); // Перенос текста, если он длинный
                    label.setMaxWidth(Double.MAX_VALUE); // Максимальная ширина для метки
                    label.setStyle("-fx-padding: 5;"); // Добавляем отступы
                    setGraphic(label); // Устанавливаем метку как графический элемент для ячейки
                }
            }
        });
        
        // Загрузка заметок при инициализации
        loadNotes();
        
        // Обработчик для отображения выбранной заметки в полях ввода
        notesListView.getSelectionModel().selectedItemProperty().addListener(
            (obs, oldVal, newVal) -> showSelectedNote(newVal));
    }

    // Метод для загрузки всех заметок из базы данных и их отображения в ListView
    private void loadNotes() {
        try {
            List<Note> noteList = noteDao.getAllNotes(); // Получаем все заметки из базы данных
            notes.setAll(noteList); // Заполняем список заметок
        } catch (SQLException e) {
            // Если возникла ошибка, показываем сообщение об ошибке
            showAlert("Ошибка базы данных", e.getMessage());
        }
    }

    // Метод для отображения выбранной заметки в полях ввода
    private void showSelectedNote(Note note) {
        if (note != null) {
            // Если заметка выбрана, показываем ее данные в полях ввода
            noteTitleField.setText(note.getTitle());
            noteContentArea.setText(note.getContent());
        } else {
            // Если заметка не выбрана, очищаем поля
            noteTitleField.clear();
            noteContentArea.clear();
        }
    }

    // Метод для добавления новой заметки
    @FXML
    private void addNote() {
        // Получаем данные, введенные пользователем
        String title = noteTitleField.getText().trim();
        String content = noteContentArea.getText().trim();
        
        // Проверка, что заголовок не пустой
        if (title.isEmpty()) {
            showAlert("Ошибка", "Введите заголовок заметки");
            return;
        }
        
        // Создание новой заметки и добавление ее в базу данных
        try {
            Note newNote = new Note(title, content); // Создаем объект заметки
            noteDao.addNote(newNote); // Добавляем заметку в базу данных
            notes.add(newNote); // Добавляем заметку в список для отображения
            clearNoteFields(); // Очищаем поля ввода
        } catch (SQLException e) {
            // Если возникла ошибка при добавлении заметки в базу, показываем ошибку
            showAlert("Ошибка базы данных", e.getMessage());
        }
    }

    // Метод для удаления выбранной заметки
    @FXML
    private void deleteNote() {
        // Получаем выбранную заметку из списка
        Note selected = notesListView.getSelectionModel().getSelectedItem();
        if (selected == null) {
            // Если заметка не выбрана, показываем ошибку
            showAlert("Ошибка", "Выберите заметку для удаления");
            return;
        }
        
        // Попытка удалить заметку из базы данных
        try {
            // Если удаление заметки успешно, удаляем ее из списка
            if (noteDao.deleteNote(selected.getId())) {
                notes.remove(selected); // Убираем заметку из отображаемого списка
                clearNoteFields(); // Очищаем поля ввода
            }
        } catch (SQLException e) {
            // Если возникла ошибка при удалении заметки, показываем ошибку
            showAlert("Ошибка базы данных", e.getMessage());
        }
    }

    // Метод для очистки полей ввода заголовка и содержания
    private void clearNoteFields() {
        noteTitleField.clear(); // Очищаем поле заголовка
        noteContentArea.clear(); // Очищаем поле содержания
    }

    // Метод для отображения сообщений об ошибках в диалоговом окне
    private void showAlert(String title, String message) {
        // Создаем диалоговое окно ошибки
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title); // Заголовок окна
        alert.setHeaderText(null); // Без дополнительного заголовка
        alert.setContentText(message); // Текст сообщения
        alert.showAndWait(); // Показываем окно и ждем, пока оно будет закрыто
    }

    // Метод для переключения на главный экран (PrimaryController)
    @FXML
    private void switchToPrimary() {
        // Переключение на главный экран приложения
        App.setRoot("primary");
    }
}
