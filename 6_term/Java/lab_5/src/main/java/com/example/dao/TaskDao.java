package com.example.dao;

import com.example.models.Task;
import com.example.Database;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.time.LocalDate; // ← ДОБАВЛЕНО
//добавляет задачу
public class TaskDao {
    public void addTask(Task task) throws SQLException {
        String sql = "INSERT INTO tasks(title, description, due_date, priority, completed) VALUES(?,?,?,?,?)";

        try (Connection conn = Database.connect();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, task.getTitle());
            pstmt.setString(2, task.getDescription());
            pstmt.setString(3, task.getDueDate().toString());
            pstmt.setInt(4, task.getPriority());

            pstmt.executeUpdate();
        }
    }
//получает задачи
    public List<Task> getAllTasks() throws SQLException {
        List<Task> tasks = new ArrayList<>();
        String sql = "SELECT * FROM tasks";

        try (Connection conn = Database.connect();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {

            while (rs.next()) {
                Task task = new Task();
                task.setId(rs.getInt("id"));
                task.setTitle(rs.getString("title"));
                task.setDescription(rs.getString("description"));
                task.setDueDate(LocalDate.parse(rs.getString("due_date"))); 
                task.setPriority(rs.getInt("priority"));

                tasks.add(task);
            }
        }

        return tasks;
    }
    public boolean deleteTask(int id) throws SQLException {
        String sql = "DELETE FROM tasks WHERE id = ?";
        
        try (Connection conn = Database.connect();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, id);
            int affectedRows = pstmt.executeUpdate();
            return affectedRows > 0;
        }
    }

}
