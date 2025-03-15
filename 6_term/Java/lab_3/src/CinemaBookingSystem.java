import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CinemaBookingSystem {

    private static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm");
    static List<Cinema> cinemas = new ArrayList<>();
    static List<Movie> movies = new ArrayList<>();
    static List<Session> sessions = new ArrayList<>();
    static Scanner input = new Scanner(System.in);

    public static void main(String[] args) {
        while (true) {
            System.out.println(
                    "Если хотите войти как администратор, нажмите 1. Если хотите войти как пользователь, нажмите 2:");
            String role = input.nextLine();
            if (role.equals("1")) {
                admin();
            } else if (role.equals("2")) {
                user();
            } else {
                System.out.println("Введите 1 или 2!");
            }
        }
    }

    // Меню администратора
    static void admin() {
        System.out.println("Введите логин администратора:");
        String username = input.nextLine();
        System.out.println("Введите пароль администратора:");
        String password = input.nextLine();

        if (!username.equals("adminLogin") || !password.equals("adminPassword")) {
            System.out.println("Неверный логин или пароль!");
            return;
        }

        while (true) {
            System.out.println("1) Добавить кинотеатр");
            System.out.println("2) Добавить зал");
            System.out.println("3) Добавить фильм");
            System.out.println("4) Создать сеанс");
            System.out.println("5) Выйти из меню администратора");
            int choice = Integer.parseInt(input.nextLine());

            switch (choice) {
                case 1:
                    addCinema();
                    break;
                case 2:
                    addHall();
                    break;
                case 3:
                    addMovie();
                    break;
                case 4:
                    createSession();
                    break;
                case 5:
                    return;
                default:
                    System.out.println("Введите цифру от 1 до 5!");
            }
        }
    }

    // Меню пользователя
    static void user() {
        System.out.println("Введите логин пользователя:");
        String username = input.nextLine();
        System.out.println("Введите пароль пользователя:");
        String password = input.nextLine();

        if (!username.equals("userLogin") || !password.equals("userPassword")) {
            System.out.println("Неверный логин или пароль!");
            return;
        }

        while (true) {
            System.out.println("1) Найти ближайший сеанс");
            System.out.println("2) Купить билет");
            System.out.println("3) Показать план зала");
            System.out.println("4) Выйти из меню пользователя");
            int choice = Integer.parseInt(input.nextLine());

            switch (choice) {
                case 1:
                    findNearestSession();
                    break;
                case 2:
                    buyTicket();
                    break;
                case 3:
                    showSeatMap();
                    break;
                case 4:
                    return;
                default:
                    System.out.println("Введите цифру от 1 до 4!");
            }
        }
    }

    // Методы для администратора
    static void addCinema() {
        System.out.println("Введите название кинотеатра:");
        String name = input.nextLine();
        cinemas.add(new Cinema(name));
    }

    static void addHall() {
        System.out.println("Введите название кинотеатра:");
        String cinemaName = input.nextLine();
        Cinema cinema = findCinema(cinemaName);
        if (cinema != null) {
            System.out.println("Введите номер зала:");
            int num = Integer.parseInt(input.nextLine());
            System.out.println("Введите количество рядов:");
            int rows = Integer.parseInt(input.nextLine());
            System.out.println("Введите количество мест в ряду:");
            int cols = Integer.parseInt(input.nextLine());
            cinema.addHall(new Hall(num, rows, cols));
        } else {
            System.out.println("Кинотеатр не найден!");
        }
    }

    static void addMovie() {
        System.out.println("Введите название фильма:");
        String title = input.nextLine();
        System.out.println("Введите продолжительность фильма (в минутах):");
        int duration = Integer.parseInt(input.nextLine());
        movies.add(new Movie(title, duration));
    }

    static void createSession() {
        System.out.println("Введите название фильма:");
        String movieTitle = input.nextLine();
        Movie movie = findMovie(movieTitle);
        if (movie != null) {
            System.out.println("Введите название кинотеатра:");
            String cinemaName = input.nextLine();
            Cinema cinema = findCinema(cinemaName);
            if (cinema != null) {
                System.out.println("Введите номер зала:");
                int hallNum = Integer.parseInt(input.nextLine());
                Hall hall = findHall(cinema, hallNum);
                if (hall != null) {
                    System.out.println("Введите время сеанса (день.месяц.год часы:минуты):");
                    String timeString = input.nextLine();
                    try {
                        LocalDateTime time = LocalDateTime.parse(timeString, DATE_TIME_FORMATTER);
                        sessions.add(new Session(movie, time, hall, cinema)); // Передаём cinema
                    } catch (DateTimeParseException e) {
                        System.out.println("Неверный формат даты и времени!");
                    }
                } else {
                    System.out.println("Зал не найден!");
                }
            } else {
                System.out.println("Кинотеатр не найден!");
            }
        } else {
            System.out.println("Фильм не найден!");
        }
    }

    static void findNearestSession() {
        System.out.println("Введите название фильма:");
        String movieTitle = input.nextLine();
        Session session = findNearestSession(movieTitle);
        if (session != null) {
            String formattedTime = session.time.format(DATE_TIME_FORMATTER);
            System.out.println(
                    "Ближайший сеанс: " + session.movie.title + formattedTime +
                            " в зале " + session.hall.num + " кинотеатра " + session.cinema.name);
        } else {
            System.out.println("Сеансы не найдены!");
        }
    }

    static void buyTicket() {
        System.out.println("Введите название фильма:");
        String movieTitle = input.nextLine();
        Session session = findNearestSession(movieTitle);
        if (session != null) {
            System.out.println("Введите ряд (от 1 до " + session.hall.rows + "):");
            int row = Integer.parseInt(input.nextLine()) - 1; // Корректируем индекс
            System.out.println("Введите место (от 1 до " + session.hall.cols + "):");
            int col = Integer.parseInt(input.nextLine()) - 1; // Корректируем индекс

            if (row >= 0 && row < session.hall.rows && col >= 0 && col < session.hall.cols) {
                Seat seat = session.hall.seats[row][col];
                if (!seat.isOccupied()) {
                    seat.occupy();
                    System.out.println("Билет куплен! Ряд " + (row + 1) + ", место " + (col + 1));
                } else {
                    System.out.println("Место уже занято!");
                }
            } else {
                System.out.println("Неверный ряд или место!");
            }
        } else {
            System.out.println("Сеансы не найдены!");
        }
    }

    static void showSeatMap() {
        System.out.println("Введите название фильма:");
        String movieTitle = input.nextLine();
        Session session = findNearestSession(movieTitle);
        if (session != null) {
            session.hall.printSeats();
        } else {
            System.out.println("Сеансы не найдены!");
        }
    }

    // Вспомогательные методы
    static Cinema findCinema(String name) {
        for (Cinema cinema : cinemas) {
            if (cinema.name.equals(name)) {
                return cinema;
            }
        }
        return null;
    }

    static Movie findMovie(String title) {
        for (Movie movie : movies) {
            if (movie.title.equals(title)) {
                return movie;
            }
        }
        return null;
    }

    static Hall findHall(Cinema cinema, int num) {
        for (Hall hall : cinema.halls) {
            if (hall.num == num) {
                return hall;
            }
        }
        return null;
    }

    static Session findNearestSession(String movieTitle) {
        LocalDateTime now = LocalDateTime.now();
        Session nearestSession = null;
        for (Session session : sessions) {
            if (session.movie.title.equals(movieTitle) && session.time.isAfter(now) && session.availableSeats()) {
                if (nearestSession == null || session.time.isBefore(nearestSession.time)) {
                    nearestSession = session;
                }
            }
        }
        return nearestSession;
    }
}
