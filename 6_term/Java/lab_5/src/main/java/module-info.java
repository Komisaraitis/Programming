module com.example {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.sql;
    requires org.xerial.sqlitejdbc;

    opens com.example to javafx.fxml;
    opens com.example.models to javafx.base;
    exports com.example;
    exports com.example.dao;
    exports com.example.models;
}