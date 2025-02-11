import java.util.Scanner;
import java.util.Arrays;

public class Main {
    static Scanner input = new Scanner(System.in);

    public static void main(String[] args) {
        System.out.println(
                "Выберите номер задания\n1)Найти наибольшую подстроку без повторяющихся символов\n2)Объединить два отсортированных массива\n3)Найти максимальную сумму подмассива\n4)Повернуть массив на 90 градусов по часовой стрелке\n5)Найти пару элементов в массиве, сумма которых равна заданному числу\n6)Найти сумму всех элементов в двумерном массиве\n7)Найти максимальный элемент в каждой строке двумерного массива\n8)Повернуть двумерный массив на 90 градусов против часовой стрелке");
        int task = input.nextInt();
        input.nextLine();
        switch (task) {
            case 1:
                substring();
                break;
            case 2:
                merging();
                break;
            case 3:
                maxSum();
                break;
            case 4:
                clockwise();
                break;
            case 5:
                sumNumber();
                break;
            case 6:
                sumArray();
                break;
            case 7:
                maxElement();
                break;
            case 8:
                counterclockwise();
                break;
        }

        input.close();
    }

    static void substring() {
        System.out.println("Введите строку:");
        String str = input.nextLine();
        int length = str.length();
        boolean[] let = new boolean[128];
        int start = 0, maxLength = 0, startInd = 0;

        char[] chars = str.toCharArray();

        for (int end = 0; end < length; end++) {
            char currChar = chars[end];

            while (let[currChar]) {
                let[chars[start]] = false;
                start++;
            }

            let[currChar] = true;

            if (end - start + 1 > maxLength) {
                maxLength = end - start + 1;
                startInd = start;
            }
        }

        System.out.println(
                "Наибольшая подстрока без повторяющихся символов: " + str.substring(startInd, startInd + maxLength));

    }

    static void merging() {
        System.out.println("Введите количество элементов первого массива:");
        int size1 = input.nextInt();
        int[] array1 = new int[size1];
        System.out.println("Введите элементы первого массива:");
        for (int i = 0; i < size1; i++) {
            array1[i] = input.nextInt();
        }

        System.out.println("Введите количество элементов второго массива:");
        int size2 = input.nextInt();
        int[] array2 = new int[size2];
        System.out.println("Введите элементы второго массива:");
        for (int i = 0; i < size2; i++) {
            array2[i] = input.nextInt();
        }

        int[] mergedArray = new int[array1.length + array2.length];

        int i = 0, j = 0, k = 0;

        while (i < array1.length || j < array2.length) {
            if (j >= array2.length || (i < array1.length && array1[i] <= array2[j])) {
                mergedArray[k++] = array1[i++];
            } else {
                mergedArray[k++] = array2[j++];
            }
        }

        System.out.println("Объединенный массив:\n" + Arrays.toString(mergedArray));

    }

    static void maxSum() {
        System.out.println("Введите количество элементов массива:");
        int size = input.nextInt();
        int[] nums = new int[size];

        System.out.println("Введите элементы массива:");
        for (int i = 0; i < size; i++) {
            nums[i] = input.nextInt();
        }

        int maxSum = Integer.MIN_VALUE;

        for (int i = 0; i < size; i++) {
            int currSum = 0;
            for (int j = i; j < size; j++) {
                currSum += nums[j];
                if (currSum > maxSum) {
                    maxSum = currSum;
                }
            }
        }

        System.out.println("Максимальная сумма подмассива(последовательных элементов): " + maxSum);

    }

    static void clockwise() {

        System.out.println("Введите количество строк массива:");
        int row = input.nextInt();
        System.out.println("Введите количество столбцов массива:");
        int col = input.nextInt();

        int[][] array = new int[row][col];

        System.out.println("Введите элементы массива:");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                array[i][j] = input.nextInt();
            }
        }

        int[][] clockwiseArray = new int[col][row];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                clockwiseArray[j][row - 1 - i] = array[i][j];
            }
        }

        System.out.println("Массив, повернутый на 90 градусов по часовой стрелке:");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                System.out.print(clockwiseArray[i][j] + " ");
            }
            System.out.println();
        }

    }

    static void sumNumber() {
        System.out.println("target:");
        int target = input.nextInt();
        System.out.println("Введите количество элементов массива:");
        int size = input.nextInt();
        int[] nums = new int[size];
        System.out.println("Введите элементы массива:");
        for (int i = 0; i < size; i++) {
            nums[i] = input.nextInt();
        }

        boolean pairFound = false;

        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) {
                    System.out.println("Пара элементов:\n" + nums[i] + ", " + nums[j]);
                    pairFound = true;
                    return;
                }
            }
        }

        if (!pairFound) {
            System.out.println("null");
        }
    }

    static void sumArray() {
        System.out.println("Введите количество строк массива:");
        int row = input.nextInt();
        System.out.println("Введите количество столбцов массива:");
        int col = input.nextInt();

        int sum = 0;

        System.out.println("Введите элементы массива:");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                sum += input.nextInt();
            }
        }
        System.out.println("Сумма всех элементов в двумерном массиве: " + sum);
    }

    static void maxElement() {
        System.out.println("Введите количество строк массива:");
        int row = input.nextInt();
        System.out.println("Введите количество столбцов массива:");
        int col = input.nextInt();

        int[][] array = new int[row][col];
        int[] maxArray = new int[row];

        System.out.println("Введите элементы массива:");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                array[i][j] = input.nextInt();
            }
            maxArray[i] = array[i][0];
            for (int j = 1; j < col; j++) {
                if (array[i][j] > maxArray[i]) {
                    maxArray[i] = array[i][j];
                }
            }
        }

        System.out.println("Максимальные элементы в каждой строке:");
        for (int max : maxArray) {
            System.out.print(max + " ");
        }
    }

    static void counterclockwise() {
        System.out.println("Введите количество строк массива:");
        int row = input.nextInt();
        System.out.println("Введите количество столбцов массива:");
        int col = input.nextInt();

        int[][] array = new int[row][col];

        System.out.println("Введите элементы массива:");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                array[i][j] = input.nextInt();
            }
        }

        int[][] counterclockwiseArray = new int[col][row];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                counterclockwiseArray[col - 1 - j][i] = array[i][j];
            }
        }

        System.out.println("Массив, повернутый на 90 градусов против часовой стрелки:");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                System.out.print(counterclockwiseArray[i][j] + " ");
            }
            System.out.println();
        }
    }
}
