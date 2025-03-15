import java.util.ArrayList;
import java.util.List;

public class Cinema {
    String name;
    List<Hall> halls = new ArrayList<>();

    Cinema(String name) {
        this.name = name;
    }

    void addHall(Hall hall) {
        halls.add(hall);
    }
}
