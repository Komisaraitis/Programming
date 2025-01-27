import java.util.Scanner;

public class Main {
    static Scanner input = new Scanner(System.in);

    public static void main(String[] args) {
        System.out.println(
                "Выберите номер задания\n1)Сиракузская последовательность \n2)Сумма ряда\n3)Ищем клад\n4)Логистический максимин\n5)Дважды четное число");
        int task = input.nextInt();
        switch (task) {
            case 1:
                syracuseSequence();
                break;
            case 2:
                rowSum();
                break;
            case 3:
                treasure();
                break;
            case 4:
                logisticMax();
                break;
            case 5:
                doubleEven();
                break;
        }

        input.close();
    }

    static void syracuseSequence() {
        int n = input.nextInt();
        int step = 0;
        while (n != 1) {
            if (n % 2 == 0) {
                n = n / 2;
            } else {
                n = 3 * n + 1;
            }
            step++;
        }
        System.out.println(step);
    }

    static void rowSum() {
        int n = input.nextInt();
        int summ = 0;
        for (int i = 1; i < n + 1; i++) {
            int num = input.nextInt();
            if (i % 2 != 0) {
                summ += num;
            } else {
                summ -= num;
            }
        }
        System.out.println(summ);
    }

    static void treasure() {
        int treasureX = input.nextInt();
        int treasureY = input.nextInt();

        int x = 0;
        int y = 0;

        int count = 0;

        int minSteps = Integer.MAX_VALUE;

        while (true) {
            String direction = input.next();
            if (direction.equals("stop")) {
                break;
            }

            int step = input.nextInt();

            switch (direction) {
                case "north":
                    y += step;
                    break;
                case "south":
                    y -= step;
                    break;
                case "west":
                    x -= step;
                    break;
                case "east":
                    x += step;
                    break;
            }

            count++;

            if (x == treasureX && y == treasureY) {
                if (count < minSteps) {
                    minSteps = count;
                }
            }
        }
        System.out.println(minSteps);

    }

    static void logisticMax() {
        int count = input.nextInt();
        int maxHeight = Integer.MIN_VALUE;
        int numRoad = 0;
        for (int i = 1; i < count + 1; i++) {
            int num = input.nextInt();
            int minHeight = Integer.MAX_VALUE;

            for (int j = 0; j < num; j++) {
                int tunnelHeight = input.nextInt();
                if (minHeight > tunnelHeight) {
                    minHeight = tunnelHeight;
                }
            }

            if (minHeight > maxHeight) {
                maxHeight = minHeight;
                numRoad = i;
            }
        }
        System.out.println(numRoad + " " + maxHeight);
    }

    static void doubleEven() {
        System.out.print("Введите трехзначное положительное число:\n");
        int num = input.nextInt();

        int hundreds = num / 100;
        int tens = (num / 10) % 10;
        int units = num % 10;

        int sum = hundreds + tens + units;
        int product = hundreds * tens * units;

        if (sum % 2 == 0 && product % 2 == 0) {
            System.out.println("дважды чётное");
        } else {
            System.out.println("не дважды чётное");
        }
    }
}
