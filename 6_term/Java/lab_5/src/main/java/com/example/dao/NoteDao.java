package com.example.dao;

import com.example.models.Note;
import com.example.Database;
import java.sql.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;

public class NoteDao {
    private static final DateTimeFormatter DATETIME_FORMATTER = 
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
//добавляет заметку
    public void addNote(Note note) throws SQLException {
        String sql = "INSERT INTO notes(title, content, created_at) VALUES(?,?,?)";
        
        try (Connection conn = Database.connect();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, note.getTitle());
            pstmt.setString(2, note.getContent());
            pstmt.setString(3, note.getCreatedAt().format(DATETIME_FORMATTER));
            pstmt.executeUpdate();
        }
    }
//получает заметки 
    public List<Note> getAllNotes() throws SQLException {
        List<Note> notes = new ArrayList<>();
        String sql = "SELECT id, title, content, created_at FROM notes";
        
        try (Connection conn = Database.connect();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            while (rs.next()) {
                Note note = new Note();
                note.setId(rs.getInt("id"));
                note.setTitle(rs.getString("title"));
                note.setContent(rs.getString("content"));
                
                String dateString = rs.getString("created_at");
                try {
                    LocalDateTime createdAt = dateString != null ? 
                        LocalDateTime.parse(dateString, DATETIME_FORMATTER) : 
                        LocalDateTime.now();
                    note.setCreatedAt(createdAt);
                } catch (DateTimeParseException e) {
                    System.err.println("Ошибка парсинга даты: " + dateString);
                    note.setCreatedAt(LocalDateTime.now());
                }
                
                notes.add(note);
            }
        }
        
        return notes;
    }
//удаляет заметку   
    public boolean deleteNote(int id) throws SQLException {
        String sql = "DELETE FROM notes WHERE id = ?";
        
        try (Connection conn = Database.connect();
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setInt(1, id);
            int affectedRows = pstmt.executeUpdate();
            return affectedRows > 0;
        }
    }
    

}