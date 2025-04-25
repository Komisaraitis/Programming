import java.util.*;

public class LibraryTest {
    public static void main(String[] args) {
        Library library = new Library();

        Book book1 = new Book("Виноваты звёзды", "Джон Грин", 2012);
        Book book2 = new Book("В поисках Аляски", "Джон Грин", 2005);
        Book book3 = new Book("Спеши любить", "Николас Спаркс", 1999);
        Book book4 = new Book("Дневник памяти", "Николас Спаркс", 1994);
        Book book5 = new Book("Отелло", "Уильям Шекспир", 1604);
        library.addBook(book1);
        library.addBook(book2);
        library.addBook(book3);
        library.addBook(book4);
        library.addBook(book5);

        System.out.println("Добавим книги:\n");
        library.printAllBooks();

        System.out.println("\nПоиск по автору:");
        System.out.println("\nКниги Джона Грина:");
        library.findBooksByAuthor("Джон Грин").forEach(System.out::println);

        System.out.println("\nПоиск по году:");
        System.out.println("\nКниги 1999 года:");
        library.findBooksByYear(1999).forEach(System.out::println);

        System.out.println("\nУдаление книги:\n");
        System.out.println("Удалим книгу'Виноваты звёзды':");
        library.removeBook(book1);
        System.out.println("После удаления 'Виноваты звёзды':");
        library.printAllBooks();

        library.printUniqueAuthors();

        library.printAuthorStatistics();
    }
}
