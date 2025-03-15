import java.time.LocalDateTime;

public class Session {
    Movie movie;
    LocalDateTime time;
    Hall hall;
    Cinema cinema; // Добавляем поле для кинотеатра

    Session(Movie movie, LocalDateTime time, Hall hall, Cinema cinema) {
        this.movie = movie;
        this.time = time;
        this.hall = hall;
        this.cinema = cinema; // Инициализируем поле
    }

    boolean availableSeats() {
        for (Seat[] row : hall.seats) {
            for (Seat seat : row) {
                if (!seat.isOccupied()) {
                    return true;
                }
            }
        }
        return false;
    }
}
