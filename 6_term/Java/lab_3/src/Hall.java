public class Hall {
    int num; // ID зала
    int rows; // Количество рядов
    int cols; // Количество мест в ряду
    Seat[][] seats; // Массив мест

    Hall(int num, int rows, int cols) {
        this.num = num;
        this.rows = rows;
        this.cols = cols;
        this.seats = new Seat[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                seats[i][j] = new Seat(i, j); // Инициализация мест
            }
        }
    }

    // Вывод схемы зала с нумерацией рядов и мест
    void printSeats() {
        System.out.println("План зала " + num + ":");
        System.out.print("    "); // Отступ для номеров мест
        for (int j = 0; j < cols; j++) {
            System.out.print((j + 1) + "   "); // Номера мест (начиная с 1)
        }
        System.out.println();

        for (int i = 0; i < rows; i++) {
            System.out.print((i + 1) + "  "); // Номер ряда (начиная с 1)
            for (int j = 0; j < cols; j++) {
                System.out.print(seats[i][j].isOccupied() ? "[X] " : "[ ] "); // Место
            }
            System.out.println();
        }
    }
}
