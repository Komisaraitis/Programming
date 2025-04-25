import java.util.Objects;

public class Book {
    private String title;
    private String author;
    private int year;

    public Book(String title, String author, int year) {
        this.title = title;
        this.author = author;
        this.year = year;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public int getYear() {
        return year;
    }

    public String toString() {
        return "Книга: " + title + ", автор: " + author + ", год: " + year;
    }

    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }

        if (!(other instanceof Book)) {
            return false;
        }

        Book otherBook = (Book) other;

        return this.year == otherBook.year
                && this.title.equals(otherBook.title)
                && this.author.equals(otherBook.author);
    }

    public int hashCode() {
        return Objects.hash(title, author, year);
    }
}
