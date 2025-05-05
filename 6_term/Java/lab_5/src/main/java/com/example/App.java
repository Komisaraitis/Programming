package com.example;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URL;

public class App extends Application {
    private static Stage primaryStage;
    private static Scene mainScene;

    // Минимальные размеры окна
    private static final int MIN_WIDTH = 800;
    private static final int MIN_HEIGHT = 600;

    @Override
    public void start(Stage stage) throws IOException {
        primaryStage = stage;

        // Загружаем главное окно
        Parent root = loadFXML("primary");

        // Создаем сцену с минимальными размерами
        mainScene = new Scene(root, MIN_WIDTH, MIN_HEIGHT);

        // Настраиваем главное окно
        setupPrimaryStage();

        // Показываем окно
        primaryStage.show();
    }

    /**
     * Загружает интерфейс из FXML-файла
     * @param fxml Название FXML-файла (без расширения .fxml)
     * @return Загруженный интерфейс как контейнер Parent
     * @throws IOException Если файл не найден или поврежден
     */
    private static Parent loadFXML(String fxml) throws IOException {
        // Ищем FXML-файл в ресурсах приложения
        URL resourceUrl = App.class.getResource(fxml + ".fxml");
        if (resourceUrl == null) {
            throw new IOException("FXML файл не найден: " + fxml + ".fxml");
        }

        // Загружаем и разбираем FXML
        FXMLLoader loader = new FXMLLoader(resourceUrl);
        try {
            return loader.load();
        } catch (IOException e) {
            System.err.println("Ошибка при чтении FXML: " + e.getMessage());
            throw new IOException("Ошибка загрузки интерфейса: " + fxml, e);
        }
    }

    /**
     * Настраивает основные параметры главного окна
     */
    private void setupPrimaryStage() {
        // Устанавливаем сцену для окна
        primaryStage.setScene(mainScene);
        primaryStage.setTitle("Менеджер задач и заметок");

        // Фиксируем минимальный размер окна
        primaryStage.setMinWidth(MIN_WIDTH);  // Минимальная ширина
        primaryStage.setMinHeight(MIN_HEIGHT); // Минимальная высота

        // Позиционируем окно по центру экрана
        primaryStage.centerOnScreen();
    }

    /**
     * Переключает текущий экран приложения
     * @param fxml Название FXML-файла нового экрана (без .fxml)
     */
    public static void setRoot(String fxml) {
        try {
            // Загружаем новый интерфейс
            Parent newRoot = loadFXML(fxml);
            
            // Устанавливаем новый интерфейс как текущий
            mainScene.setRoot(newRoot);
            
            // Подгоняем размер окна под содержимое
            primaryStage.sizeToScene();
        } catch (IOException e) {
            // Показываем ошибку, если загрузка не удалась
            showErrorAlert("Ошибка переключения", 
                        "Не удалось загрузить: " + fxml + "\nПричина: " + e.getMessage());
        }
    }

    /**
     * Показывает всплывающее окно с сообщением об ошибке
     * @param title Заголовок окна
     * @message Текст сообщения об ошибке
     */
    public static void showErrorAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.initOwner(primaryStage);  // Привязываем к главному окну
        alert.setTitle(title);
        alert.setHeaderText(null);      // Скрываем дополнительный заголовок
        alert.setContentText(message);  // Устанавливаем основной текст
        alert.showAndWait();           // Показываем и ждем закрытия
    }

    /**
     * Возвращает текущую сцену приложения
     */
    public static Scene getMainScene() {
        return mainScene;
    }

    /**
     * Возвращает главное окно приложения
     */
    public static Stage getPrimaryStage() {
        return primaryStage;
    }

    /**
     * Точка входа в приложение
     */
    public static void main(String[] args) {
        // Подготавливаем базу данных (создаем таблицы)
        Database.initialize();
        
        // Запускаем JavaFX-приложение
        launch(args);
    }
}