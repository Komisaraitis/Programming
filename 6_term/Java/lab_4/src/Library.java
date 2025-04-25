import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Library {
    private List<Book> books;
    private Set<String> authors;
    private Map<String, Integer> authorStatistics;

    public Library() {
        this.books = new ArrayList<Book>();
        this.authors = new HashSet<String>();
        this.authorStatistics = new HashMap<String, Integer>();
    }

    public void addBook(Book book) {

        books.add(book);
        String author = book.getAuthor();
        authors.add(author);
        if (authorStatistics.containsKey(author)) {
            int currentCount = authorStatistics.get(author);
            authorStatistics.put(author, currentCount + 1);
        } else {
            authorStatistics.put(author, 1);
        }
    }

    public void removeBook(Book book) {

        boolean wasRemoved = books.remove(book);
        if (wasRemoved) {
            String author = book.getAuthor();
            int currentCount = authorStatistics.get(author);
            if (currentCount == 1) {
                authorStatistics.remove(author);
                authors.remove(author);
            } else {
                authorStatistics.put(author, currentCount - 1);
            }
        } else {
            System.out.println("Книга не найдена в библиотеке: " + book);
        }
    }

    public List<Book> findBooksByAuthor(String author) {

        List<Book> booksByAuthor = new ArrayList<Book>();
        for (Book book : books) {
            if (author.equals(book.getAuthor())) {
                booksByAuthor.add(book);
            }
        }
        return booksByAuthor;
    }

    public List<Book> findBooksByYear(int year) {
        List<Book> booksByYear = new ArrayList<Book>();
        for (Book book : books) {
            if (year == book.getYear()) {
                booksByYear.add(book);
            }
        }
        return booksByYear;
    }

    public void printAllBooks() {

        System.out.println("Список всех книг в библиотеке:");
        for (Book book : books) {
            System.out.println(book);
        }
    }

    public void printUniqueAuthors() {

        System.out.println("\nСписок уникальных авторов:\n");
        for (String author : authors) {
            System.out.println(author);
        }
    }

    public void printAuthorStatistics() {

        System.out.println("\nСтатистика по авторам:\n");
        for (Map.Entry<String, Integer> entry : authorStatistics.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue() + " книг(и)");
        }
    }
}
