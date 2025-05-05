package com.example;

import com.example.dao.TaskDao;
import com.example.models.Task;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import java.io.IOException;
import java.sql.SQLException;
import java.time.LocalDate;
import java.util.List;

public class PrimaryController {

    // Элементы UI, привязанные к компонентам в FXML
    @FXML
    private TableView<Task> taskTable; // Таблица для отображения задач
    @FXML
    private TableColumn<Task, String> titleColumn; // Столбец для заголовков задач
    @FXML
    private TableColumn<Task, String> dueDateColumn; // Столбец для дат выполнения
    @FXML
    private TableColumn<Task, String> descriptionColumn; // Столбец для описаний задач
    @FXML
    private TableColumn<Task, Number> priorityColumn; // Столбец для приоритетов задач

    @FXML
    private TextField taskTitleField; // Поле для ввода заголовка задачи
    @FXML
    private TextArea taskDescriptionArea; // Поле для ввода описания задачи
    @FXML
    private DatePicker dueDatePicker; // Компонент для выбора даты выполнения
    @FXML
    private ComboBox<Integer> priorityComboBox; // Комбинированный список для выбора приоритета задачи

    // DAO-объект для работы с задачами в базе данных
    private final TaskDao taskDao = new TaskDao();

    // Метод инициализации, вызываемый при старте контроллера
    @FXML
    public void initialize() {
        // Настройка автоматического изменения размера столбцов таблицы
        taskTable.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY);

        // Настройка отображения данных в столбцах таблицы
        titleColumn.setCellValueFactory(cellData -> cellData.getValue().titleProperty());
        dueDateColumn.setCellValueFactory(cellData -> cellData.getValue().dueDateStringProperty());
        descriptionColumn.setCellValueFactory(cellData -> cellData.getValue().descriptionProperty());
        priorityColumn.setCellValueFactory(cellData -> cellData.getValue().priorityProperty());

        // Заполнение ComboBox значениями приоритетов от 1 до 5
        priorityComboBox.setItems(FXCollections.observableArrayList(1, 2, 3, 4, 5));

        // Загрузка задач из базы данных при инициализации
        try {
            loadTasks();
        } catch (SQLException e) {
            e.printStackTrace();
            // Показ ошибки, если не удалось загрузить задачи
            showAlert("Ошибка базы данных", e.getMessage());
        }
    }

    // Метод для загрузки всех задач из базы данных и отображения их в таблице
    private void loadTasks() throws SQLException {
        // Получаем список задач из базы данных
        List<Task> tasks = taskDao.getAllTasks();
        // Заполняем таблицу данными
        taskTable.setItems(FXCollections.observableArrayList(tasks));
    }

    // Метод для добавления новой задачи
    @FXML
    private void addTask() {
        // Получаем значения, введенные пользователем
        String title = taskTitleField.getText().trim();
        String description = taskDescriptionArea.getText().trim();
        LocalDate dueDate = dueDatePicker.getValue();
        Integer priority = priorityComboBox.getValue();

        // Проверка, что все необходимые поля заполнены
        if (title.isEmpty() || dueDate == null || priority == null) {
            showAlert("Ошибка", "Пожалуйста, заполните все поля");
            return;
        }

        // Создание новой задачи и добавление её в базу данных
        try {
            Task task = new Task(title, description, dueDate, priority);
            taskDao.addTask(task); // Добавляем задачу в базу данных
            loadTasks(); // Перезагружаем список задач из базы
            clearTaskFields(); // Очищаем поля ввода
        } catch (SQLException e) {
            e.printStackTrace();
            // Показ ошибки, если возникла проблема при добавлении задачи
            showAlert("Ошибка базы данных", e.getMessage());
        }
    }

    // Метод для удаления выбранной задачи
    @FXML 
    private void deleteTask() {
        // Получаем выбранную задачу из таблицы
        Task selected = taskTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            // Если задача не выбрана, показываем сообщение об ошибке
            showAlert("Ошибка", "Выберите задачу для удаления");
            return;
        }
        
        // Попытка удалить задачу из базы данных
        try {
            if (taskDao.deleteTask(selected.getId())) {
                // Если удаление успешно, удаляем задачу из таблицы
                taskTable.getItems().remove(selected);
            }
        } catch (SQLException e) {
            // Показ ошибки, если возникла проблема при удалении задачи
            showAlert("Ошибка базы данных", e.getMessage());
        }
    }

    // Метод для очистки всех полей ввода задачи
    private void clearTaskFields() {
        taskTitleField.clear(); // Очищаем поле для заголовка
        taskDescriptionArea.clear(); // Очищаем поле для описания
        dueDatePicker.setValue(null); // Сбрасываем выбранную дату
        priorityComboBox.setValue(null); // Сбрасываем выбранный приоритет
    }

    // Метод для отображения сообщений об ошибках в диалоговом окне
    private void showAlert(String title, String message) {
        // Создаем и показываем Alert с типом ERROR
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title); // Устанавливаем заголовок
        alert.setHeaderText(null); // Убираем заголовок
        alert.setContentText(message); // Устанавливаем текст сообщения
        alert.showAndWait(); // Показываем alert и ждем закрытия
    }

    // Метод для переключения на вторичный экран (вторичную сцену)
    @FXML
    private void switchToSecondary() throws IOException {
        // Переключаем на вторичный экран через контроллер App
        App.setRoot("secondary");
    }
}
