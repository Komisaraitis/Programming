public class Seat {
    int row;
    int col;
    boolean occupied = false; // Ленивая инициализация

    Seat(int row, int col) {
        this.row = row;
        this.col = col;
    }

    boolean isOccupied() {
        return occupied;
    }

    void occupy() {
        this.occupied = true;
    }
}
